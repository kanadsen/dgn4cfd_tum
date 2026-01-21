
import vtk
def vtk_convert_sample(fname, fout, reader=vtk.vtkGenericEnSightReader(), writer=vtk.vtkXMLMultiBlockDataWriter()):
    reader.SetCaseFileName(fname)
    reader.Update()
    
    writer.SetFileName(fout)
    writer.SetInputConnection(reader.GetOutputPort());
    writer.Update();

#vtk_convert_sample(,'/home/kanadsen01/Desktop/Git_repos/Forked_Repos/dgn4cfd_tum/data/solution/MURI_slab-fluid.vtk', reader=vtk.vtkGenericEnSightReader(), writer=vtk.vtkXMLMultiBlockDataWriter())


import os
import zipfile

root_dir = "/lus/grand/projects/NeuralDE/kanadsen/dataset_trial2"        # Directory you want to scan (recursively)
extract_root = "/lus/grand/projects/NeuralDE/kanadsen/dataset_trial2"   # Where extracted folders will be created

os.makedirs(extract_root, exist_ok=True)

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".encas"):
            zip_path = os.path.join(dirpath, filename)
            vtk_convert_sample(zip_path,zip_path, reader=vtk.vtkGenericEnSightReader(), writer=vtk.vtkXMLMultiBlockDataWriter())
    print(f"Finished extracting {filename}")
print("Finished extracting all zip files.")