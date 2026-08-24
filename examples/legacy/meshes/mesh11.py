show_figures = True
save_mesh_to_file = True

#%%
# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-paper"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

#%%
# =============================================================================
# Geometry
# =============================================================================
g = cfg.Geometry()

# points
g.point([-400,355])     # 0
g.point([-400,-100])    # 1
g.point([1365,-100])    # 2
g.point([1365,110])     # 3
g.point([1365,130])     # 4
g.point([1150,140])     # 5
g.point([1120,148])     # 6
g.point([1080,148])     # 7
g.point([1070,152])     # 8
g.point([1050,152])     # 9
g.point([280,360])      # 10

g.point([40,280])       # 11
g.point([120,250])      # 12
g.point([300,250])      # 13
g.point([340,230])      # 14
g.point([360,225])      # 15
g.point([410,200])      # 16
g.point([460,190])      # 17
g.point([510,200])      # 18
g.point([1080,120])     # 19
g.point([1230,110])     # 20

# lines
g.line([0,1])   # 1
g.line([1,2])   # 2
g.line([2,3])   # 3
g.line([3,4])   # 4
g.line([4,5])   # 5
g.line([5,6])   # 6
g.line([6,7])   # 7
g.line([7,8])   # 8
g.line([8,9])   # 9
g.line([9,10])  # 10
g.line([10,0])  # 11
g.line([0,11])  # 12
g.line([11,10]) # 13
g.line([11,12]) # 14
g.line([12,13]) # 15
g.line([13,14]) # 16
g.line([14,15]) # 17
g.line([15,16]) # 18
g.line([16,17]) # 19
g.line([17,18]) # 20
g.line([18,19]) # 21
g.line([19,20]) # 22
g.line([20,3])  # 23

# surfaces
g.surface([10,11,12])   # 0
g.surface([3,4,5,6,7,8,9,12,13,14,15,16,17,18,19,20,21,22])     # 1
g.surface([0,1,2,22,21,20,19,18,17,16,15,14,13,11])  # 2



#%%
# =============================================================================
# Mesh
# =============================================================================
mesh = cfm.GmshMesh(g,el_size_factor=20)

coords, edof, dofs, bdofs, elementmarkers = mesh.create()
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

#%% Plots
if show_figures:
    # geometry plot
    plt.figure()
    cfv.draw_geometry(g)

    # mesh plot
    plt.figure()
    cfv.draw_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True
    )
    plt.title(f"Mesh")
    plt.suptitle(f"el_size_factor={mesh.el_size_factor}, N={coords.shape[0]} nodes", fontsize=8, y=0.90)

    plt.show()

