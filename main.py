import streamlit as st
import random
import matplotlib.pyplot as plt
import pandas as pd
import imbalanced_databases as imbd
import os
from knnor import data_augment

sampled_datasets_below_1000=[
    imbd.load_ecoli_0_1_4_6_vs_5(),
    imbd.load_ecoli3(),
    imbd.load_glass6(),
    imbd.load_new_thyroid1(),
]

@st.cache
def get_augmented_data(filename):
    print("Filename changed, getting augmented data")
    this_data = pd.read_csv(filename)    
    return this_data

@st.cache
def get_original_data(filename):
    print("Filename changed, getting original data")
    for i in range(len(sampled_datasets_below_1000)):
        dataset=sampled_datasets_below_1000[i]    
        fname=dataset['DESCR']
        if fname==filename:
            # returna a pandas version of this file
            # two dataframes, first is the minority
            # second is the majority
            X=dataset["data"]
            y=dataset["target"]
            knnor = data_augment.KNNOR()
            min_label,min_index=knnor.get_minority_label_index(X,y)
            X_min=X[y==min_label]
            X_maj=X[y!=min_label]

            # now make the dataframe
            count_cols=X.shape[1]
            columns=[]
            for i in range(count_cols):
                columns.append("feat"+str(i))
            df_min = pd.DataFrame(X_min, columns = columns)
            df_maj = pd.DataFrame(X_maj, columns = columns)

            return df_min, df_maj






header=st.container()
choose_dataset = st.container()
dataset = st.container()


with header:
    st.header("Welcome to a visual representation of KNNOR")


with choose_dataset:
    st.header("Please choose the dataset")
    list_files=os.listdir("data")
    list_files=[i.split(".csv")[0] for i in list_files]
    print(list_files)
    file_name=st.selectbox('Select File', options=list_files, index = 0)
    print("File name selected is ",file_name)



with dataset:    
    st.header(file_name)
    st.text('I found this dataset on UCB')
    

    aug_data = get_augmented_data('data/'+file_name+'.csv')
    print("Getting the data")
    print(aug_data.isnull().sum().sum())
    min_df,maj_df=get_original_data(file_name)

    feature_names=[]
    for col in aug_data.columns:
        if "feat" in col:
            feature_names.append(col)
    uniq_props=list(aug_data.proportion.unique())
    print("Unique peoportions are ",uniq_props)
    sel_col, disp_col = st.columns(2)
    proportion = sel_col.selectbox('What is the proportion?', options=uniq_props, index = 0)

    num_nbrs_max=int(aug_data.num_neighbors.max())
    num_nbrs_min=int(aug_data.num_neighbors.min())

    num_nbrs = sel_col.slider('Number of neighbors?', min_value=num_nbrs_min, 
                            max_value=num_nbrs_max, value=num_nbrs_min, step=2)

    max_dist_max=float(aug_data.max_dist.max())
    max_dist_min=float(aug_data.max_dist.min())
    print(max_dist_min,max_dist_max)

    max_dist = sel_col.slider('Distance', 
        min_value=max_dist_min, max_value=max_dist_max, 
        value=max_dist_min, step=0.2)

    # going for proportion of minority data points used
    min_prop_minority=float(aug_data.prop_minority.min())
    max_prop_minority=float(aug_data.prop_minority.max())
    print("Proportions of minority used",min_prop_minority,"to",max_prop_minority)
    prop_minority_used=sel_col.slider("Proportion of minority",
        min_value=min_prop_minority, max_value=max_prop_minority, 
        value=min_prop_minority, step=0.2)

        
    #feature 1
    feat1=sel_col.selectbox('Feature for axis 1?', options=feature_names, index = 0)
    
    new_feature_reversed = list(reversed(feature_names))
    feat2=sel_col.selectbox('Feature for axis 2?', options=new_feature_reversed, index = 0)
    print("LOOking for proportion ",proportion)
    print("Looking for num neighbors",num_nbrs)

    disp_col.subheader('Output scatter:')


    # filter happens here
    aug_data=aug_data[aug_data["proportion"]==proportion]  
    print("1 factoring proportion of min pop after augmentation",aug_data.shape)
    aug_data=aug_data[aug_data["num_neighbors"]==num_nbrs]  
    print("2 factoring neighbors",aug_data.shape)
    aug_data=aug_data[aug_data["max_dist"]==max_dist]  
    print("3 factoring max distance",aug_data.shape)
    aug_data=aug_data[aug_data["prop_minority"]==prop_minority_used]  
    print("4 factoring proportion of minority used",aug_data.shape)


    print(feat1,feat2)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # the augmened data
    x1s=aug_data[feat1]
    x2s=aug_data[feat2]
    # print("X1s",x1s)
    # print("X2s",x2s)    
    plt.scatter(x1s,x2s,color='green',marker="*",label="artificial", alpha=1)

    x1s=min_df[feat1]
    x2s=min_df[feat2]
    plt.scatter(x1s,x2s,color='red',label='minority', alpha=0.3)

    x1s=maj_df[feat1]
    x2s=maj_df[feat2]
    plt.scatter(x1s,x2s,color='orange',label='majority', alpha=0.2)

    plt.legend()

    

    
    
    disp_col.write(fig)
    disp_col.caption('Count of majority datapoints:')
    disp_col.write(maj_df.shape[0])

    disp_col.caption('Count of original minority datapoints:')
    disp_col.write(min_df.shape[0])
    
    disp_col.caption('Count of augmented minority datapoints:')
    disp_col.write(aug_data.shape[0])
    
