import glob
import subprocess

for filein in glob.glob("meshseg_orig/*.off"):
    print(filein)
    fileout = "meshseg/{}".format(filein.split("/")[1])
    subprocess.call(["meshlabserver", "-i", filein, "-o", fileout, "-s", "simplify-mesh.mlx"])
