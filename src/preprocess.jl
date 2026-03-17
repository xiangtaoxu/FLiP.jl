"""
    preprocess(pc::PointCloud; cfg::FLiPConfig=_CFG) -> PointCloud

Prepare point cloud before ground segmentation:
1. Optional distance-based subsampling
2. Optional statistical filtering
"""
function preprocess(pc::PointCloud; cfg::FLiPConfig=_CFG)
    active = cfg.pipeline_enable_subsample ? distance_subsample(pc, cfg.pipeline_subsample_res) : pc

    if cfg.preprocess_enable_statistical_filter
        active = statistical_filter(
            active,
            cfg.statistical_filter_k_neighbors,
            cfg.statistical_filter_n_sigma,
        )
    end

    return active
end
