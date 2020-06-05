import matplotlib.pyplot as plt
import numpy as np

# plt.plot([0,60],[0,60],label='old')
# plt.plot([0,760],[0,760],label='new')
# plt.bar([10],['old'],align='center',alpha=.5)
# plt.show()

def draw_bar_chart(names,values,barPath,title='species',rotation='0',save=False,show=True):
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, names, rotation=rotation)
    plt.ylabel('images number')
    plt.title(title)
    if save:
        plt.savefig(barPath)
    if show:
        plt.show()


objects = ('old','new')*8
y_pos = np.arange(len(objects))
performance = [60,800]*8

draw_bar_chart(objects,performance,'bar.png',save=True,show=True)
