import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import io
import imageio
from textwrap import wrap
import torch
import moviepy.editor as mpe
from scipy.io import wavfile 
import os

def plot_3d_motion(out_path, joints, kinematic_chain, title=None, ground=True, figsize=(10, 10), fps=120):
    matplotlib.use('Agg')
    
    data = joints.copy().reshape(len(joints), -1, 3)
    frame_number = data.shape[0]
    
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    
    def update(index):

        def init():
            ax.set_xlim3d([-0.8, 0.8])
            ax.set_ylim3d([0, 1.6])
            ax.set_zlim3d([0, 1.6])
            ax.grid(False)
        
        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)
        
        fig = plt.figure(figsize=figsize, dpi=96)
        ax = fig.add_subplot(111, projection='3d')
        
        if title is not None :
            wraped_title = '\n'.join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16)
        
        ax.view_init(elev=130, azim=-90)
        init()
        
        # ax.cla()

        if ground:
            plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                        MAXS[2] - trajec[index, 1])
            if index > 1:
                ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                        trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                        color='blue')


        for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        
            # ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=0,
            #             color=color, marker="o", markersize=linewidth*1.5, markerfacecolor="g", markeredgecolor="g")

        # for i in range(data[index].shape[0]):
        #     ax.text(data[index][i][0], data[index][i][1], data[index][i][2], str(i))
            
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', dpi=96)
        io_buf.seek(0)
        arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        plt.close()
        return arr

    out = []
    for i in range(frame_number) : 
        out.append(update(i))
    out = np.stack(out, axis=0)
    out = np.array(torch.from_numpy(out))
    imageio.mimsave(out_path, out, fps=fps)


def add_audio(out_path, audio_paths):
    filename, ext = os.path.splitext(out_path)
    in_path = filename + "_tmp" + ext
    os.system(f"cp {out_path} {in_path}")
    my_clip = mpe.VideoFileClip(in_path)
    if len(audio_paths) > 1:
        audio_clips = []
        for path in audio_paths:
            audio_clips.append(mpe.AudioFileClip(path))
        final_audio = mpe.concatenate_audio(audio_clips)
    else:
        final_audio = mpe.AudioFileClip(audio_paths[0])
    final_clip = my_clip.set_audio(final_audio)
    final_clip.write_videofile(out_path)
    # os.system(f'rm -f {in_path}')

def get_audio_length(audio_path):
    sample_rate, data = wavfile.read(audio_path) 
    len_data = len(data) 
    t = len_data / sample_rate  # duration in floats 
    return t