import pandas as pd 
import numpy as np 
from trustopinion import TrustOpinion 
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
matplotlib.rcParams.update({'font.size': 16})

DEBUG = False

def compute_opinion(class_value, representatives, half_step, data):
    dic = {} #Clusters
    res = {}
    arr_l = data["True Label"]
    # arr = data[data["True Label"]==curr_class][f"Class_{curr_class}_Probability"]
    arr = data[f"Class_{class_value}_Probability"]
    for representative in representatives:
        n_i = np.where( (arr>=representative-half_step) & (arr<representative+half_step))
        dic[representative] = n_i[0].tolist() 
    
    all_pos = 0
    all_neg = 0
    for representative in representatives:
        tmp = arr_l[dic[representative]] # All the data in the current cluster
        t_c = np.sum(tmp == class_value) # Number of true data in this cluster
        n_c = len(tmp) # Number of data in this cluster

        # We want tc/nc close to representative 
        # tc/nc - repr prop to tc - nc*rep

        pos_ev = t_c
        neg_ev = np.round(np.abs(t_c-n_c*representative))
        res[representative] = (pos_ev, neg_ev)
        all_pos+=pos_ev
        all_neg+=neg_ev
    # print(res)
    if(DEBUG):
        print(all_pos, all_neg, all_neg+all_pos)
    resOp1 = TrustOpinion.ev2tdu(all_pos, all_neg)
    resOp2 = None 
    resOp3 = None
    val = list(res.values())
    if(len(representatives)>1):
        resOp2 = TrustOpinion.ev2tdu(val[0][0], val[0][1])
        for i in range(1, len(representatives)):
            resOp2 = TrustOpinion.cumFuse(resOp2, TrustOpinion.ev2tdu(val[i][0], val[i][1]))
    if(len(representatives)>1):
        resOp3 = TrustOpinion.ev2tdu(val[0][0], val[0][1])
        for i in range(1, len(representatives)):
            resOp3 = TrustOpinion.avFuse(resOp3, TrustOpinion.ev2tdu(val[i][0], val[i][1]))
    return res, resOp1, resOp2, resOp3
    # return res, resOp2, resOp1, resOp3

def main():
    n_clusters = 10
    step = 1/n_clusters
    first_representative = step/2
    representatives = [round(first_representative+i*step , 5) for i in range(n_clusters)]
    if(DEBUG):   
        print(representatives)
    # ## PARAMS
    # directory = "MNIST_PRED_T"
    # epochs = list(range(1,10))+list(range(10, 101, 10))

    directory = "CIFAR10_L"
    epochs = [1]+list(range(10, 101, 10))
    Tos_bef = {}
    Tos_aft = {}
    
    ## PARAMS

    # directory = "MNIST_PRED_T"
    # epochs = list(range(1,10))+list(range(10, 101, 10))
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    # print(files)
    for curr_class in range(10):
        Tos_aft[f"{curr_class}"] = []
        Tos_bef[f"{curr_class}"] = []
        res_dic = {}
        for f in files:
            if(DEBUG):
                print("----------------------------------start----------------------------------")
                print(f)
            data = pd.read_csv(f)
            res, resOp1, resOp2, resOp3 = compute_opinion(curr_class, representatives,first_representative, data)
            # print(res)
            # print(resOp1)
            # print(resOp2)
            # print(resOp3)
            # res_dic[f] = (resOp1.t, resOp1.d, resOp1.u)
            res_dic[f] = resOp1
        # print(res_dic)
        
    
        for epoch in epochs:
            key_aft = f"{directory}\\aft_{epoch}.csv"
            key_bef = f"{directory}\\bef_{epoch}.csv"

           
            Tos_aft[f"{curr_class}"].append(res_dic[key_aft])
            Tos_bef[f"{curr_class}"].append(res_dic[key_bef])
        # for epoch in range(10, 101, 10):
        # # for epoch in range(10, 51, 10):
        #     key_aft = f"{directory}\\aft_{epoch}.csv"
        #     key_bef = f"{directory}\\bef_{epoch}.csv"
        #     Tos_aft.append(res_dic[key_aft])
        #     Tos_bef.append(res_dic[key_bef])
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 4)) 
    # fig, axs = plt.subplots(1, 3, figsize=(12, 4)) 

    # epochs = list(range(1, 10))+list(range(10, 101, 10))
    
    # epochs = list(range(1, 10))+list(range(10, 51, 10))
    colors = cm.viridis(np.linspace(0, 1, 10))
    for lab in range(10):
        axs[1][0].plot(epochs, [op.t for op in Tos_aft[str(lab)]], color=colors[lab])
        axs[0][0].plot(epochs, [op.t for op in Tos_bef[str(lab)]], color=colors[lab])
        axs[1][1].plot(epochs, [op.d for op in Tos_aft[str(lab)]], color=colors[lab])
        axs[0][1].plot(epochs, [op.d for op in Tos_bef[str(lab)]], color=colors[lab])
        axs[1][2].plot(epochs, [op.u for op in Tos_aft[str(lab)]], color=colors[lab])
        axs[0][2].plot(epochs, [op.u for op in Tos_bef[str(lab)]], color=colors[lab])


        # axs[0].plot(epochs, [op.t for op in Tos_aft[str(lab)]], color=colors[lab], alpha=1/(lab+1))
        # axs[0].plot(epochs, [op.t for op in Tos_bef[str(lab)]], color=colors[10+lab], alpha=1/(lab+1))
        # axs[1].plot(epochs, [op.d for op in Tos_aft[str(lab)]], color=colors[lab], alpha=1/(lab+1))
        # axs[1].plot(epochs, [op.d for op in Tos_bef[str(lab)]], color=colors[10+lab], alpha=1/(lab+1))
        # axs[2].plot(epochs, [op.u for op in Tos_aft[str(lab)]], color=colors[lab], alpha=1/(lab+1))
        # axs[2].plot(epochs, [op.u for op in Tos_bef[str(lab)]], color=colors[10+lab], alpha=1/(lab+1))

        # axs[0].plot(epochs, [op.t for op in Tos_aft[str(lab)]], label=f"After Temperature Scaling {lab}")
        # axs[0].plot(epochs, [op.t for op in Tos_bef[str(lab)]], label=f"Before Temperature Scaling {lab}")
        # axs[1].plot(epochs, [op.d for op in Tos_aft[str(lab)]], label=f"After Temperature Scaling {lab}")
        # axs[1].plot(epochs, [op.d for op in Tos_bef[str(lab)]], label=f"Before Temperature Scaling {lab}")
        # axs[2].plot(epochs, [op.u for op in Tos_aft[str(lab)]], label=f"After Temperature Scaling {lab}")
        # axs[2].plot(epochs, [op.u for op in Tos_bef[str(lab)]], label=f"Before Temperature Scaling {lab}")
    

    # axs[0].legend(loc='lower left')
    # axs[0].set_xlabel("Epochs")
    # axs[0].set_ylabel("Trust")
    

    
    # axs[1].legend(loc='upper left')
    # axs[1].set_xlabel("Epochs")
    # axs[1].set_ylabel("DisTrust")

    
    # axs[2].legend(loc='upper left')
    # axs[2].set_xlabel("Epochs")
    # axs[2].set_ylabel("Uncertainty")
    # # axs[2].gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 100:.1f}'))
    # axs[2].yaxis.set_major_formatter(ticker.ScalarFormatter())
    # axs[2].ticklabel_format(axis='y', style='sci', scilimits=(-2,-2))
    # sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=1, vmax=10))

    # axs[1][0].legend(loc='lower left')
    axs[1][0].set_xlabel("Epochs")
    axs[1][0].set_ylabel("Trust")
    axs[0][0].set_ylabel("Trust")

    axs[1][1].set_xlabel("Epochs")
    axs[1][1].set_ylabel("DisTrust")
    axs[0][1].set_ylabel("DisTrust")

    axs[1][2].set_xlabel("Epochs")
    axs[1][2].set_ylabel("Uncertainty")
    axs[0][2].set_ylabel("Uncertainty")

    # axs[0][2].yaxis.set_major_formatter(ticker.ScalarFormatter())
    axs[0][2].ticklabel_format(axis='y', style='sci', scilimits=(-2,-2))
    # axs[1][2].yaxis.set_major_formatter(ticker.ScalarFormatter())
    axs[1][2].ticklabel_format(axis='y', style='sci', scilimits=(-2,-2))

    for i in range(2):
        ylim1 = axs[0][i].get_ylim()   # Get y-axis limits from the first subplot in the first row
        ylim2 = axs[1][i].get_ylim()   # Get y-axis limits from the first subplot in the first row
        

        # Apply the same y-axis limits and ticks to the corresponding subplot in the second row
        
        
        ylim=(min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))

        axs[1][i].set_ylim(ylim)
        axs[0][i].set_ylim(ylim)
        # yticks = axs[0][i].get_yticks()  # Get y-axis ticks from the first subplot in the first row
        # axs[1][i].set_yticks(yticks)
        yticks = np.linspace(ylim[0], ylim[1], 5)
        yticks = np.around(yticks, 1) 
        axs[1][i].set_yticks(yticks)
        axs[0][i].set_yticks(yticks)
        # print(ylim)
    i=2
    ylim1 = axs[0][i].get_ylim()   # Get y-axis limits from the first subplot in the first row
    ylim2 = axs[1][i].get_ylim()   # Get y-axis limits from the first subplot in the first row
    

    # Apply the same y-axis limits and ticks to the corresponding subplot in the second row
    
    
    ylim=(min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))

    axs[1][i].set_ylim(ylim)
    axs[0][i].set_ylim(ylim)
    # yticks = axs[0][i].get_yticks()  # Get y-axis ticks from the first subplot in the first row
    # axs[1][i].set_yticks(yticks)
    yticks = np.linspace(ylim[0], ylim[1], 5)
    yticks = np.around(yticks, 3) 
    axs[1][i].set_yticks(yticks)
    axs[0][i].set_yticks(yticks)
    # print(ylim)
    # axs[0][1].set_title("Before Calibration")

    # axs[1].legend(loc='upper left')
    # axs[1].set_xlabel("Epochs")
    # axs[1].set_ylabel("DisTrust")

    
    # axs[2].legend(loc='upper left')
    # axs[2].set_xlabel("Epochs")
    # axs[2].set_ylabel("Uncertainty")
    # # axs[2].gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 100:.1f}'))
    # axs[2].yaxis.set_major_formatter(ticker.ScalarFormatter())
    # axs[2].ticklabel_format(axis='y', style='sci', scilimits=(-2,-2))


    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=1, vmax=10))


    sm.set_array([])  # Only needed for older versions of matplotlib
    # cbar = fig.colorbar(sm)
    # cbar = fig.colorbar(sm)
    # cbar = fig.colorbar(sm, ax=axs, location='right', fraction=0.05, pad=0.04)
    fig.subplots_adjust(right=0.95)  # Adjust space on the right for colorbar

# Create an axis for the colorbar on the right side
    cbar_ax = fig.add_axes([0.97, 0.15, 0.005, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    # cbar.set_label('Label')
    fig.text(0.5, 0.90, "Before Calibration", ha='center', fontsize=16)  # Title for the first row
    fig.text(0.5, 0.465, "After Calibration", ha='center', fontsize=16) 
    # plt.tight_layout()
    line = plt.Line2D([0.05, 0.95], [0.49, 0.49], color='black', linewidth=1.5, transform=fig.transFigure)
    fig.add_artist(line)
    line2 = plt.Line2D([0.05, 0.95], [0.93, 0.93], color='black', linewidth=1.5, transform=fig.transFigure)
    fig.add_artist(line2)
    plt.show()

def main_2():
    n_clusters = 10
    step = 1/n_clusters
    first_representative = step/2
    representatives = [round(first_representative+i*step , 5) for i in range(n_clusters)]
    if(DEBUG):   
        print(representatives)
    directory = "CIFAR10_PRED"
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    # print(files)
    res_dic = {}
    for f in files:
        if(DEBUG):
            print("----------------------------------start----------------------------------")
            print(f)
        data = pd.read_csv(f)
        curr_class = 5
        res, resOp1, resOp2, resOp3 = compute_opinion(curr_class, representatives,first_representative, data)
        break 
    print(res)
    print(resOp1)
    print(resOp2)
    print(resOp3)
main()
# print(data)


# dic = {}

# arr_l = data["True Label"]
# # arr = data[data["True Label"]==curr_class][f"Class_{curr_class}_Probability"]
# arr = data[f"Class_{curr_class}_Probability"]
# print(arr)
# n = len(arr)
# n2 = 0
# # for i in range(n):

# for representative in representatives:
#     n_i = np.where( (arr>=representative-first_representative) & (arr<representative+first_representative))
#     dic[representative] = n_i[0].tolist() 


# print(n_i)
# print(arr[n_i])
#     # n2+=n_i
# for representative in representatives:
#     tmp = arr_l[dic[representative]]
#     t_c = np.sum(tmp == curr_class)
#     n_c = len(tmp)
#     print(representative, t_c, n_c, t_c/n_c)
#     # abs(n_c*r_c - t_c)




    





