import shapely

import dagster as dg
import geopandas as gpd
import numpy as np

from scipy.spatial.distance import jensenshannon
from sklearn_extra.cluster import KMedoids
from urban_shape.partitions import zone_partitions
from urban_shape.resources import MedoidConfigResource


def polygons_factory(top_prefix: str) -> dg.AssetsDefinition:
    @dg.asset(
        name="base",
        key_prefix=[top_prefix, "polygons"],
        ins={"df_points": dg.AssetIn([top_prefix, "points", "remove_unused"])},
        partitions_def=zone_partitions,
        io_manager_key="geodataframe_manager",
        group_name=f"polygons_{top_prefix}",
    )
    def _asset(df_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        hulls = {}
        for label, subdf in df_points.groupby("label"):
            if label == -1:
                continue

            coords = subdf.geometry.get_coordinates().to_numpy()
            if len(coords) < 3:
                hulls[label] = None
            else:
                points = shapely.geometry.MultiPoint(coords)
                hulls[label] = shapely.concave_hull(points, ratio=0.1)
                
        return gpd.GeoDataFrame(gpd.GeoSeries(hulls, crs=df_points.crs).reset_index()).rename(columns={"index": "label"})

    return _asset


def polygons_labeled_factory(top_prefix: str) -> dg.AssetsDefinition:
    @dg.asset(
        name="labeled",
        key_prefix=[top_prefix, "polygons"],
        ins={"df_polygons": dg.AssetIn([top_prefix, "polygons", "base"]), "df_points": dg.AssetIn([top_prefix, "points", "remove_unused"])},
        partitions_def=zone_partitions,
        required_resource_keys=set([f"{top_prefix}_medoid_config_resource"]),
        io_manager_key="geodataframe_manager",
        group_name=f"polygons_{top_prefix}",
    )
    def _asset(context: dg.AssetExecutionContext, df_polygons: gpd.GeoDataFrame, df_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        config: MedoidConfigResource = getattr(context.resources, f"{top_prefix}_medoid_config_resource")        

        df_cross = (
            df_points
            .assign(category=lambda df: df["codigo_act"].astype(str).str[:2].astype(int))
            .groupby(["label", "category"])
            ["codigo_act"]
            .count()
            .reset_index()
            .pivot_table(index="label", columns="category", values="codigo_act")
            .fillna(0)
            .astype(int)
        )
        labels = df_cross.index.tolist()
        
        mat = df_cross.to_numpy()
        mat = mat / mat.sum(axis=1)[:, np.newaxis]

        dist_mat = np.zeros((len(mat), len(mat)))
        for i, start_row in enumerate(mat):
            for j, end_row in enumerate(mat):
                dist_mat[i, j] = jensenshannon(start_row, end_row)

        model = KMedoids(config.n_clusters, metric="precomputed").fit(dist_mat)
        cluster_label_to_medoid_label_map = {key: value for key, value in zip(labels, model.labels_, strict=True)}

        df_polygons["medoid"] = df_polygons["label"].map(cluster_label_to_medoid_label_map)
        return df_polygons                

    return _asset

dassets = [factory(top_prefix) for factory in [polygons_factory, polygons_labeled_factory] for top_prefix in ["coarse", "fine"]]