# %% 

import pandas as pd
from datetime import datetime
from neurometry import viz, cave, ic
import matplotlib.pyplot as plt
import numpy as np

set_one = pd.read_csv("outputs/set_one_synapses_11_02_2022_14:20:45.csv")
set_two = pd.read_csv("outputs/set_two_synapses_11_02_2022_14:20:45.csv")
set_three = pd.read_csv("outputs/set_three_synapses_11_02_2022_14:20:45.csv")

total = pd.concat([set_one, set_two, set_three])

#client = cave.set_up()
#vol = cave.get_cloud_volume('precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig_sharded')


# %%
source = ic.source()
id_to_location = {}
id_to_presyn = {}
id_to_postsyn = {}
num = 1
for_df = []
for index, row in total.iterrows():
    parsed = row["pre_pt_position"][1:-1].split(" ")
    numbers = []
    for x in parsed:
        if x != "":
            numbers.append(int(x))
    #print(numbers)
    tupled = tuple(numbers)
    tupled = (int(tupled[0]//4), int(tupled[1]//4), int(tupled[2]//40))
    id_to_location[row["id"]] = tupled
    id_to_presyn[row["id"]] = row["pre_pt_root_id"]
    id_to_postsyn[row["id"]] = row["post_pt_root_id"]

    bounds = ic.create_bounds(id_to_location[row["id"]],320, 16)
    imgvol, segdict = ic.get_img_and_seg(source, bounds, id_to_presyn[row["id"]], id_to_postsyn[row["id"]])
    #overlay = ic.get_overlay(imgvol, segdict)
    #ic.stacker(overlay)
    for_df.append(["dcvsyn" + str(num), row["id"], tupled])
    ic.writer("cutouts/dcvsyn" + str(num) + "/img/vol.tiff", np.transpose(imgvol))
    ic.writer("cutouts/dcvsyn" + str(num) + "/presyn/presyn.tiff", np.transpose(segdict[id_to_presyn[row["id"]]]*255).astype('uint8'))
    ic.writer("cutouts/dcvsyn" + str(num) + "/postsyn/postsyn.tiff", np.transpose(segdict[id_to_postsyn[row["id"]]]*255).astype('uint8'))
    num += 1
# %%
dcv_df = pd.DataFrame(for_df, columns=["wk_id", "syn_id", "coordinates"])
dcv_df.to_csv("outputs/dcvsyn600.csv", index=False)
# A lot of this afterwards is scratch code
# %%
bounds = ic.create_bounds(id_to_location[112178450],
                            320, 16)
imgvol, segdict = ic.get_img_and_seg(source, bounds, 
id_to_presyn[112178450], id_to_postsyn[112178450])
overlay = ic.get_overlay(imgvol, segdict)
ic.stacker(overlay)

# %%
bounds = ic.create_bounds(id_to_location[176637500],
                            320, 16)
imgvol, segdict = ic.get_img_and_seg(source, bounds, 
id_to_presyn[176637500], id_to_postsyn[176637500])
overlay = ic.get_overlay(imgvol, segdict)
ic.stacker(overlay)

# %%
bounds = ic.create_bounds(id_to_location[122583367],
                            320, 16)
imgvol, segdict = ic.get_img_and_seg(source, bounds, 
id_to_presyn[122583367], id_to_postsyn[122583367])
overlay = ic.get_overlay(imgvol, segdict)
ic.stacker(overlay)

# %% 
bounds = ic.create_bounds(id_to_location[111544163],
                            320, 16)
imgvol, segdict = ic.get_img_and_seg(source, bounds, 
id_to_presyn[111544163], id_to_postsyn[111544163])
overlay = ic.get_overlay(imgvol, segdict)
ic.stacker(overlay)

# %% 
bounds = ic.create_bounds(id_to_location[18995809],
                            320, 16)
imgvol, segdict = ic.get_img_and_seg(source, bounds, 
id_to_presyn[18995809], id_to_postsyn[18995809])
overlay = ic.get_overlay(imgvol, segdict)
ic.stacker(overlay)
# %%
import PIL
cutout = cave.get_cutout(id_to_location[111148410], vol, 250, 250, 20)

# %%
#the white thing is not part of the image
import numpy as np
PIL.Image.fromarray(cutout[:,:,10])

# %%
segmentation = cave.get_cloud_volume('graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31')

# %%
elements = segmentation.download((slice(38308-125,38308+125), slice(17612-125,17612+125), slice(1010,1030)), agglomerate=True)
vcs = list(elements.reshape(-1,1))
# %%
vc_example = segmentation.download((slice(38308-125,38308+125), slice(17612-125,17612+125), slice(1010,1030)),agglomerate=False, segids=720575940621715550, preserve_zeros=False)[:,:,:,0]

# %%
def get_presyn_mask(tupl, seg, xsize, ysize, zsize, segid):
    x = tupl[0]
    y = tupl[1]
    z = tupl[2]
    x = x/4
    y = y/4
    xsize = xsize/2
    ysize = ysize/2
    zsize = zsize/2
    img = seg.download((slice(x-xsize,x+xsize), slice(y-ysize,y+ysize), slice(z-zsize,z+zsize)), agglomerate=False,
                        segids=segid, preserve_zeros=False, mip=0)
    
    img = img > 1
    img = img.astype(int)
    raw = np.array(img[:,:,:,0]*255, dtype=np.uint8)
    #counterclockwise rotation by 90 degrees
    rotated = np.rot90(raw).copy()
    #flip vertical
    flipped = np.flipud(rotated).copy()
    return flipped

def get_presyn_mask_two(tupl, seg, xsize, ysize, zsize, segid):
    x = tupl[0]
    y = tupl[1]
    z = tupl[2]
    #x = x/4
    #y = y/4
    xsize = xsize/2
    ysize = ysize/2
    zsize = zsize/2
    img = seg.download((slice(x-xsize,x+xsize), slice(y-ysize,y+ysize), slice(z-zsize,z+zsize)), agglomerate=False,
                        segids=segid, preserve_zeros=False, coord_resolution=(4,4,40))
    
    img = img > 1
    img = img.astype(int)
    raw = np.array(img[:,:,:,0]*255, dtype=np.uint8)
    #counterclockwise rotation by 90 degrees
    rotated = np.rot90(raw).copy()
    #flip vertical
    flipped = np.flipud(rotated).copy()
    return flipped

#get cutout in flywire space
def get_presyn_mask_three(tupl, vol, xsize, ysize, zsize, segid):
    x = tupl[0]
    y = tupl[1]
    z = tupl[2]
    img = vol.download_point(
        (x, y, z), # point in neuroglancer 
        size=(xsize, ysize, zsize),
        coord_resolution=(4,4,40),
        agglomerate=False,
        segids=segid, preserve_zeros=False

         # neuroglancer display resolution
    )
    img = img > 1
    img = img.astype(int)
    raw = np.array(img[:,:,:,0]*255, dtype=np.uint8)
    #counterclockwise rotation by 90 degrees
    rotated = np.rot90(raw).copy()
    #flip vertical
    flipped = np.flipud(rotated).copy()
    return flipped
# %% 
import matplotlib.pyplot as plt
def plot_two_images(x, y, _slice):
    f, axarr = plt.subplots(1,2)
    print(axarr)
    axarr[0].imshow(x[:,:,_slice], cmap="gray")
    axarr[1].imshow(y[:,:,_slice], cmap="gray")

# %% 
#mask=get_presyn_mask(id_to_location[111148410], segmentation, 250, 250, 20, 720575940621715550)

mask=get_presyn_mask(id_to_location[111148410], segmentation, 250, 250, 20, 720575940621715550)

# %%
plot_two_images(cutout, mask, 10)

# %%
mask_two=get_presyn_mask_two(id_to_location[111148410], segmentation, 250, 250, 20, 720575940621715550)

# %%
plot_two_images(cutout, mask_two, 9)

# %%
mask_three = get_presyn_mask_three(id_to_location[111148410], segmentation, 250, 250, 20, 720575940621715550)

# %%
plot_two_images(cutout, mask_two, 9)

# %%
plot_two_images(mask_two, mask_three, 9)




# %%
plot_two_images(cutout, mask_three, 9)

# %%
plot_two_images(mask, mask_three, 9)

# %%

import fafbseg
svs = fafbseg.flywire.roots_to_supervoxels(720575940621715550)[720575940621715550]

# %%
import matplotlib.pyplot as plt
the_cutout = np.squeeze(vol[153234-125:153234+125,70450-125:70450+125, 1020])
plt.imshow(the_cutout, cmap=plt.cm.gray)

# %% 
the_mask = np.squeeze(segmentation[153234//4 - 125:153234//4+125,
                                    70450//4 - 125:70450//4+125,
                                    1020])

# %%
#the_mask.reshape(250*250)
difference = list(set(the_mask.tolist()).intersection(set(svs)))

# %%
the_mask_two = np.squeeze(segmentation[153234//4 - 125:153234//4+125,
                                    70450//4 - 125:70450//4+125,
                                    1020])

new_mask = np.in1d(np.array(the_mask_two), difference)
new_mask = new_mask.astype(int)
pd.Series(new_mask).value_counts()
plt.imshow(new_mask.reshape(250,250))
# %%
from imageryclient import ImageryClient

# %%
ic=ImageryClient(image_source = 'precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14' ,
                 segmentation_source='graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31',
                 image_mip=1, base_resolution=[4,4,40])

#ic=ImageryClient(image_source = 'precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig_sharded' ,
#                 segmentation_source='precomputed://https://seungdata.princeton.edu/'
#                                'sseung-archive/fafbv14-ws/'
#                                'ws_190410_FAFB_v02_ws_size_threshold_200')

#ic=ImageryClient(image_source = 'precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14',
 #                segmentation_source='graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31')
# %%
bounds=[[153234-160, 70450-160, 1020-8],
        [153234+160, 70450+160, 1020+8]]

# %%
imgvol, segdict = ic.image_and_segmentation_cutout(bounds,
                                                   split_segmentations=True, root_ids=[720575940621715550, 
                                                   720575940631304978])

# %%
pre_syn = 720575940621715550
post_syn = 720575940631304978

f , ax = plt.subplots(10,3, figsize=(10,20))
# lets loop over z sections
for i in range(10):
    # plot the images in column 0
    ax[i, 0].imshow(np.squeeze(imgvol[:,:,i]),
                    cmap=plt.cm.gray,
                    vmax=255,
                    vmin=0)
    # plot the pre-synaptic mask in column 1
    ax[i, 1].imshow(np.squeeze(segdict[pre_syn][:,:,i]))
    # plot the post-synaptic mask in column 2
    ax[i, 2].imshow(np.squeeze(segdict[post_syn][:,:,i]))
f.tight_layout()

# %%
import imageryclient as ic
overlays = ic.composite_overlay(segdict, imagery=imgvol)

# %%
stacked = np.stack(overlays)

# %%
import tifffile
tifffile.imwrite("figures/syn_example.tiff", data=stacked)

# %%
