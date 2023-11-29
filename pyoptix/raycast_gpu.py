import os
import optix as ox
import cupy as cp
import numpy as np
import ctypes
import trimesh
import torch

script_dir = os.path.dirname(__file__)
cuda_src = os.path.join(script_dir, "cuda", "raycasting_mesh.cu")

def log_callback(level, tag, msg):
    print("[{:>2}][{:>12}]: {}".format(level, tag, msg))
    pass

class OptixRayMeshIntersector():
    def __init__(
            self,
            mesh,
            validation_mode=True,
    ):
        super().__init__()
        self.ctx = ox.DeviceContext(validation_mode=validation_mode, log_callback_function=log_callback, log_callback_level=3)
        self.gas = self.create_acceleration_structure(mesh)
        self.pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                                    num_payload_values=4,
                                                    num_attribute_values=2,
                                                    exception_flags=ox.ExceptionFlags.NONE,
                                                    pipeline_launch_params_variable_name="params")

        self.module = self.create_module()
        self.program_grps = self.create_program_groups()
        self.pipeline = self.create_pipeline()
        self.sbt = self.create_sbt()
    
    def create_acceleration_structure(self, mesh):
        vertices = cp.array(mesh.vertices.data, dtype=np.float32)
        faces = cp.array(mesh.faces.data, dtype=np.uint32)
        build_input = ox.BuildInputTriangleArray(vertex_buffers=vertices, index_buffer=faces, flags=[ox.GeometryFlags.NONE])
        gas = ox.AccelerationStructure(self.ctx, build_input, compact=True)
        return gas



    def create_module(self):
        compile_opts = ox.ModuleCompileOptions(debug_level=ox.CompileDebugLevel.FULL, opt_level=ox.CompileOptimizationLevel.LEVEL_0)
        module = ox.Module(context=self.ctx, src=cuda_src, module_compile_options=compile_opts, pipeline_compile_options=self.pipeline_options)
        return module


    def create_program_groups(self):
        raygen_grp = ox.ProgramGroup.create_raygen(self.ctx, self.module, "__raygen__from_buffer")
        miss_grp = ox.ProgramGroup.create_miss(self.ctx, self.module, "__miss__buffer_miss")
        hit_grp = ox.ProgramGroup.create_hitgroup(self.ctx, self.module,
                                                entry_function_CH="__closesthit__buffer_hit")

        return raygen_grp, miss_grp, hit_grp


    def create_pipeline(self):
        link_opts = ox.PipelineLinkOptions(max_trace_depth=1,
                                        debug_level=ox.CompileDebugLevel.FULL)

        pipeline = ox.Pipeline(self.ctx,
                            compile_options=self.pipeline_options,
                            link_options=link_opts,
                            program_groups=self.program_grps)

        pipeline.compute_stack_sizes(1,  # max_trace_depth
                                    0,  # max_cc_depth
                                    1)  # max_dc_depth

        return pipeline


    def create_sbt(self):
        raygen_grp, miss_grp, hit_grp = self.program_grps

        raygen_sbt = ox.SbtRecord(raygen_grp)
        miss_sbt = ox.SbtRecord(miss_grp)
        hit_sbt = ox.SbtRecord(hit_grp)
        sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt, hitgroup_records=hit_sbt)

        return sbt



    def launch_pipeline(self, origins, dirs, tmins, tmaxs):
        n_rays = origins.shape[0]
        hits = torch.zeros(n_rays, 4, device=origins.device)
        params_tmp = [
            ( 'u8', 'origins'),
            ( 'u8', 'dirs'),
            ( 'u8', 'tmins'),
            ( 'u8', 'tmaxs'),
            ( 'u8', 'hits'),
            ( 'u8', 'trav_handle')
        ]
        params = ox.LaunchParamsRecord(names=[p[1] for p in params_tmp],
                                    formats=[p[0] for p in params_tmp])

        params['origins'] = origins.data_ptr()
        params['dirs'] = dirs.data_ptr()
        params['tmins'] = tmins.data_ptr()
        params['tmaxs'] = tmaxs.data_ptr()
        params['hits'] = hits.data_ptr()
        params['trav_handle'] = self.gas.handle

        stream = cp.cuda.Stream()

        self.pipeline.launch(self.sbt, dimensions=(n_rays,), params=params, stream=stream)
        stream.synchronize()
        return hits
    
    def query_intersection(self, origins, dirs, tmin=0., tmax=1e16):
        n_rays = origins.shape[0] 
        device = origins.device
        tmins = torch.ones(n_rays, device=device) * tmin
        tmaxs = torch.ones(n_rays, device=device) * tmax
        return self.launch_pipeline(origins, dirs, tmins, tmaxs)

if __name__ == "__main__":
    
    mesh = trimesh.load("./mesh.ply")
    
    intersector = OptixRayMeshIntersector(mesh)
    
    n_rays = 768*1024
    origins = np.zeros(n_rays, dtype=np.dtype("3f4"))
    origins[:] = [0., 0.2, 5.]
    # origins = cp.asarray(origins)
    origins = torch.tensor(origins).cuda()
    
    dirs = np.zeros(n_rays, dtype=np.dtype("3f4"))
    dirs[:] = [0., 0., -1.]
    # dirs = cp.asarray(dirs)
    dirs = torch.tensor(dirs).cuda()
    
    hits = intersector.query_intersection(origins, dirs, 0, 6)
    print(hits)
    # hit_locs = hits[:,:3]
    # print(hits, hit_locs)
    # intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    # locations, index_ray, index_tri = intersector.intersects_location(origins.get(), dirs.get(), multiple_hits=False)
    # print(locations)
    # import pdb; pdb.set_trace()
    # print(hits)


