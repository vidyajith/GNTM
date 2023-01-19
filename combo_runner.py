import os

print(os.getcwd())
corpus=int(input("Select corpus \n 1.TMN \n 2.News20 \n 3.Reuters \n"))

if corpus==1:
    dataset='TMN'
    addr='C:/Users/rbw19/OneDrive/Desktop/GNTM/data/tmn3/'
    choice=int(input("TMN selected \n Do you want to keep the defaults 1(yes) 0(no,enter your values"))
    if choice==1:
        freq_thres=5
        edge_thres=3
        word_length_threshold=2
        num_of_topic=5
    else:
        freq_thres=int(input("Enter frequency threshold :"))
        edge_thres=int(input("Enter edge threshold :"))
        word_length_threshold=int(input("Enter custom word length threshold default is 2"))
        num_of_topic=int(input("enter the number of topics (default is 5):"))

elif corpus==2:
    dataset='News20'
    addr='C:/Users/rbw19/OneDrive/Desktop/GNTM/data/20news/'
    choice=int(input("News 20 selected \n Do you want to keep the defaults 1(yes) 0(no,enter your values"))
    if choice==1:
        freq_thres=20
        edge_thres=10
        word_length_threshold=2
        num_of_topic=5
        
    else:
        freq_thres=int(input("Enter frequency threshold : "))
        edge_thres=int(input("Enter edge threshold : "))
        word_length_threshold=int(input("Enter custom word length threshold default is 2 :"))
        num_of_topic=int(input("enter the number of topics (default is 5) :"))

elif corpus==3:
    dataset='Reuters'
    addr='C:/Users/rbw19/OneDrive/Desktop/GNTM/data/reuters/'
    choice=int(input("Reuters selcted \n Do you want to keep the defaults 1(yes) 0(no,enter your values) :"))
    if choice==1:
        freq_thres=10
        edge_thres=15
        word_length_threshold=2
        num_of_topic=5
    else:
        freq_thres=int(input("Enter frequency threshold : "))
        edge_thres=int(input("Enter edge threshold : "))
        word_length_threshold=int(input("Enter custom word length threshold default is 2 : "))
        num_of_topic=int(input("enter the number of topics (default is 5) :"))

else:
    raise InterruptedError("select Appropriate option")

print("####### Performing preprocessing !")
os.chdir(os.getcwd()+"\\dataPrepare")
os.system(f"python -W ignore preprocess.py --dataset={dataset} --word_len_threshold={word_length_threshold} --address={addr} --freq_threshold={freq_thres}")
print("preprocessing Completed ! \n ####### Graph Data creation started")
os.system(f"python -W ignore graph_data.py --dataset={dataset} --edge_threshold={edge_thres} --address={addr}")
print("preprocessing Completed ! \nGraph Data creation Completed \n####### Executing Main")
os.chdir("../")
os.system(f"python -W ignore main.py --dataset={dataset} --num_topic={num_of_topic}")




