import matplotlib
import matplotlib.cm as cm

# color_list = [
#     '#f1c40f',  # Yellow
#     '#e67e22',  # Pink
#     '#9b59b6',  # Blue
#     '#2980b9',  # Cyan
#     '#1abc9c',  # Purple
#     '#27ae60',  # Green
# ]

marker_list = [
    'o',  # circle
    'v',  # triangle_down
    '^',  # triangle_up
    's',  # square
    'D',  # diamond
    'P',  # plus
    '*',  # star
]

colormap_name = 'tab20'
cmap = cm.get_cmap('tab20')
color_list = [[cmap.colors[i * 2], cmap.colors[i * 2 + 1]] for i in range(len(cmap.colors) // 2)]

linestyle_list = [
    'solid',
    'dashed',
    'dotted',
    'dashdot',
    'densely dashdotdotted'
]

# if __name__ == '__main__':




