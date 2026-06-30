@testset "pipeline integration: tree_segmentation → qsm" begin
    # Two vertical trunks of differing size (asymmetric ⇒ deterministic tree ordering),
    # each a connected cluster of ring points climbing in z. AGH = z.
    function trunk(x0; n_z=40, r=0.08, zmax=2.0)
        pts = Float64[]
        for k in 0:n_z-1
            z = zmax * k / (n_z - 1)
            for j in 0:7
                θ = 2π * j / 8
                push!(pts, x0 + r*cos(θ), r*sin(θ), z)
            end
        end
        return collect(reshape(pts, 3, :)')
    end

    function build_cloud()
        coords = vcat(trunk(0.0; n_z=40), trunk(1.5; n_z=30))   # different sizes
        agh = coords[:, 3]
        make_test_pointcloud(coords; attrs=Dict(:AGH => agh))
    end

    function run_once(outdir)
        cfg = deepcopy(FLiP._CFG)
        cfg.pipeline.subsample_res = 0.05
        cfg.pipeline.output_dir    = outdir
        cfg.pipeline.output_prefix = "itest"
        cfg.tree.refine.enable = true
        tres = FLiP.tree_segmentation(build_cloud(); cfg=cfg)
        m = FLiP.model_nbs(pc=tres.pc_output, cfg=cfg, group_attr=:tree_nbs_id,
                           node_id_attr=:node_id, emit_surface=true)
        bm = FLiP.write_biometrics(m.nodes, cfg; output_dir=outdir, output_prefix="itest")
        return tres, m, bm
    end

    outdir1 = mktempdir()
    tres, m, bm = run_once(outdir1)
    pc = tres.pc_output

    @testset "tree cloud attributes" begin
        @test FLiP.npoints(pc) > 0
        for a in (:nbs_id, :tree_id, :tree_nbs_id)
            @test hasattribute(pc, a)
        end
        # The single persisted node id comes from the final QSM; no legacy :qsm_node_id,
        # and the trial id was dropped (non-debug run).
        @test hasattribute(pc, :node_id)
        @test !hasattribute(pc, :qsm_node_id)
        @test !hasattribute(pc, :trial_node_id)
        # at least one real tree assembled
        @test maximum(getattribute(pc, :tree_id)) >= 1
    end

    @testset "final QSM produced nodes" begin
        @test m.status == :success
        @test length(m.nodes) > 0
        @test isfile(bm.node_csv_path)
    end

    @testset "determinism: identical re-run → byte-identical node CSV" begin
        outdir2 = mktempdir()
        _, _, bm2 = run_once(outdir2)
        @test read(bm.node_csv_path) == read(bm2.node_csv_path)
    end
end
