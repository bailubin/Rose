import torch_geometric
from osgeo import gdal
from utils import *
import geopandas as gpd
from PIL import Image
import torch
import clip
import torch.nn.functional as F
import os

imgw, imgh = 224, 224
rs_base_dir=r'../data/rsimgs'
osm_base_dir=r'../data/osms'
save_dir=r'../data/graphs224code_road'
city_names=['singapore']
# city_names=['singapore50buffer','singapore70buffer', 'singapore90buffer']
for city in city_names:
    osm_dir=os.path.join(osm_base_dir, 'singapore')
    rs_dir=os.path.join(rs_base_dir, 'singapore')
    # osm_dir = os.path.join(osm_base_dir, city)
    # rs_dir=os.path.join(rs_base_dir, city)

    rsdataset = gdal.Open(os.path.join(rs_dir, 'singapore' + '_resample.tif'))
    amenity_rtree, amenity_list=get_rtree(os.path.join(osm_dir, 'amenity.shp'))
    building_rtree, building_list=get_rtree(os.path.join(osm_dir, 'building.shp'))
    landuse_rtree, landuse_list=get_rtree(os.path.join(osm_dir, 'landuse.shp'))
    leisure_rtree, leisure_list=get_rtree(os.path.join(osm_dir, 'leisure.shp'))
    nature_rtree, nature_list=get_rtree(os.path.join(osm_dir, 'nature.shp'))
    road_rtree, road_list=get_rtree(os.path.join(osm_dir, 'road_p1.shp'))

    classes = ['amenity', 'building', 'landuse', 'leisure', 'nature', 'road_p1']
    names = []
    for c in classes:
        df = gpd.read_file(os.path.join(osm_dir, c + '.shp'))
        f_tag = df['f_tag'].iloc[0]
        names.append(np.unique(df[f_tag]))
    names = np.concatenate(names)
    name_dict = {}
    for i, n in enumerate(names):
        name_dict[n] = i

    geotransform=rsdataset.GetGeoTransform()
    model,_=clip.load("ViT-B/16", device='cpu')

    node_dir = os.path.join(save_dir, 'node')
    edge_dir = os.path.join(save_dir, 'edge')
    edgew_dir = os.path.join(save_dir,'edge_w')
    img_dir=os.path.join(save_dir, 'img')
    shp_dir=os.path.join(save_dir, 'node_shp')

    width, height = rsdataset.RasterXSize, rsdataset.RasterYSize
    col_left = 0
    row_top = 0


    while row_top + imgh < height:
        col_left = 0
        while col_left + imgw < width:
            col_right = col_left + imgw
            row_bottom = row_top + imgh
            lng_left, lat_top = pixel2coord(col_left, row_top, geotransform)
            lng_right, lat_bottom = pixel2coord(col_right, row_bottom, geotransform)
            cregion = box(lng_left, lat_bottom, lng_right, lat_top, ccw=False)
            camenity_geo, camenity_feat = get_current_shp(cregion, amenity_rtree, amenity_list)
            cbuilding_geo, cbuilding_feat = get_current_shp(cregion, building_rtree, building_list)
            clanduse_geo, clanduse_feat = get_current_shp(cregion, landuse_rtree, landuse_list)
            cleisure_geo, cleisure_feat = get_current_shp(cregion, leisure_rtree, leisure_list)
            cnature_geo, cnature_feat = get_current_shp(cregion, nature_rtree, nature_list)
            croad_geo, croad_feat = get_current_shp(cregion, road_rtree, road_list)

            all_geo = np.concatenate([camenity_geo, cbuilding_geo, clanduse_geo, cleisure_geo, cnature_geo, croad_geo])
            if all_geo.shape[0] > 0:
                print(col_left, row_top)

                all_feat = camenity_feat + cbuilding_feat + clanduse_feat + cleisure_feat + cnature_feat + croad_feat
                cregion_area = cregion.area
                cregion_length = cregion.length
                all_area = np.array([x.area for x in all_geo]) / cregion_area
                all_length = np.array([x.length for x in all_geo]) / cregion_length
                all_center = np.array([coord2pixel(x.centroid.coords[0][0],
                                                   x.centroid.coords[0][1], geotransform) for x in all_geo]) - np.array([col_left, row_top])
                all_ring_degree = np.array([x.area / shapely.minimum_bounding_circle(x).area for x in all_geo])
                all_oriented_env = np.array([
                    [coord2pixel(c[0], c[1], geotransform) for c in x.oriented_envelope.boundary.coords[:4]] for x in
                    all_geo]) - np.array([col_left, row_top])
                all_tag = [x['properties']['f_tag'] for x in all_feat]
                all_label = np.array([x['properties'][tag] for x, tag in zip(all_feat, all_tag)])
                all_label_code=np.array([name_dict[x] for x in all_label])

                # 获得空间关系矩阵,disjoint 1, toches 2, overlaps/crosses 3, contains 4, within 5, equals 0, other 6
                r_m = get_spatial_relation(all_geo, all_feat)

                # 获得节点特征
                with torch.no_grad():
                    label_text = clip.tokenize(all_label)
                    label_features = model.encode_text(label_text)
                area_features = torch.unsqueeze(torch.from_numpy(all_area), dim=1)
                length_features = torch.unsqueeze(torch.from_numpy(all_length), dim=1)
                env_features=torch.from_numpy(all_oriented_env)/torch.tensor([imgw,imgh])
                env_features=env_features.flatten(start_dim=1)
                # ring_degree_features = torch.unsqueeze(torch.from_numpy(all_ring_degree), dim=1)
                pos_features = torch.from_numpy(all_center)/torch.tensor([imgw,imgh])
                all_label_code=torch.unsqueeze(torch.from_numpy(all_label_code), dim=1)

                node_features = torch.concat(
                    [label_features, area_features, length_features, env_features, pos_features, all_label_code], dim=1)

                # 获得边的特征
                edge_o, edge_d, edge_w = [], [], []
                for i in range(r_m.shape[0]):
                    for j in range(r_m.shape[1]):
                        if i != j and r_m[i, j] != 0 and r_m[i, j] != 6 and r_m[i,j]!=1:
                            edge_o.append(i)
                            edge_d.append(j)
                            edge_w.append(r_m[i, j])
                edge = torch.tensor([edge_o, edge_d])
                edge_w = torch.unsqueeze(torch.tensor(edge_w), dim=1)

                torch.save(node_features, os.path.join(node_dir, '{}_{}_{}.tensor'.format(city, col_left, row_top)))
                torch.save(edge, os.path.join(edge_dir, '{}_{}_{}.tensor'.format(city, col_left, row_top)))
                torch.save(edge_w, os.path.join(edgew_dir, '{}_{}_{}.tensor'.format(city, col_left, row_top)))


                res_df = gpd.GeoDataFrame(geometry=all_geo, crs=df.crs)
                res_df['label'] = all_label
                # print(res_df['geometry'].type)
                res_df.to_file(os.path.join(shp_dir, '{}_{}_{}.shp'.format(city, col_left, row_top)))

                img=Image.fromarray(rsdataset.ReadAsArray(col_left, row_top, imgw, imgh)[:3].transpose(1, 2, 0))
                img.save(os.path.join(img_dir, '{}_{}_{}.png'.format(city, col_left, row_top)))
            col_left += imgw
        row_top += imgh