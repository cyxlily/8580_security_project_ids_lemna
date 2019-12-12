import random
import matplotlib  
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np

def draw_deduction(mat):
    fea_list=[3,4,5,6,7]
    method_name = ['lemna','random']
    for i,method in enumerate(mat):
        plt.plot(fea_list,method,"x-",label=method_name[i])  
        plt.scatter(fea_list,method,marker='x',s=30) 
    plt.xlabel('Nfeature')
    plt.ylabel('PCR(%)')
    plt.title("Intrusion Detection")
    plt.legend()
    plt.savefig('feature_deduction_ids.png')
    plt.close()
def draw_synthetic(mat):
    fea_list=[3,4,5,6,7]
    method_name = ['lemna','random']
    for i,method in enumerate(mat):
        plt.plot(fea_list,method,"x-",label=method_name[i])  
        plt.scatter(fea_list,method,marker='x',s=30) 
    plt.xlabel('Nfeature')
    plt.ylabel('PCR(%)')
    plt.title("Intrusion Detection")
    plt.legend()
    plt.savefig('feature_synthetic_ids.png')
    plt.close()
def draw_augmentation(mat):
    fea_list=[3,4,5,6,7]
    method_name = ['lemna','random']
    for i,method in enumerate(mat):
        plt.plot(fea_list,method,"x-",label=method_name[i])  
        plt.scatter(fea_list,method,marker='x',s=30) 
    plt.xlabel('Nfeature')
    plt.ylabel('PCR(%)')
    plt.title("Intrusion Detection")
    plt.legend()
    plt.savefig('feature_augmentation_ids.png')
    plt.close()

if __name__ == '__main__':
#lemna pos 3,4,5,6,7
#random pos 3,4,5,6,7
    mat1 = [[23.5216819974,5.25624178712,3.67936925099,2.89093298292,0.788436268068],[62.9434954008,31.011826544,17.6084099869,11.9579500657,8.54139290407]]
    draw_deduction(mat1)

#lemna new 3,4,5,6,7
#random new 3,4,5,6,7
    mat2 = [[11.5637319317,11.6951379763,11.9579500657,11.826544021,78.3180026281],[2.89093298292,5.91327201051,6.43889618922,8.27858081472,10.5124835742]]
    draw_synthetic(mat2)

#lemna neg 3,4,5,6,7
#random neg 3,4,5,6,7
    mat3 = [[11.5637319317,11.6951379763,11.9579500657,11.826544021,78.3180026281],[2.89093298292,5.25624178712,7.62155059133,8.27858081472,10.2496714849]]
    draw_augmentation(mat3)