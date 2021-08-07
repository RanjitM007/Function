
#create a function to find k in knn 
def find_k(k, data, labels):
    #sort the data
    data = sorted(data, key=lambda x: x[0])
    #create a list to store the labels
    labels_list = []
    #create a list to store the distance
    distance_list = []
    #loop through the data
    for i in range(len(data)):
        #calculate the distance
        distance = 0
        for j in range(len(data[i])):
            distance += pow(data[i][j] - data[k][j], 2)
        distance = pow(distance, 0.5)
        #add the distance to the list
        distance_list.append([distance, labels[i]])
    #sort the distance
    distance_list = sorted(distance_list, key=lambda x: x[0])
    #loop through the distance
    for i in range(k):
        #add the label to the list
        labels_list.append(distance_list[i][1])
    #return the label
    return labels_list

##create a function for model cross validation
def model_cross_validation(data, labels, k):
    #create a list to store the accuracy
    accuracy_list = []
    #loop through the k
    for i in range(k):
        #find the labels
        labels_list = find_k(i, data, labels)
        #calculate the accuracy
        correct = 0
        for j in range(len(labels)):
            if labels[j] == labels_list[j]:
                correct += 1
        accuracy = correct / len(labels)
        #add the accuracy to the list
        accuracy_list.append(accuracy)
    #return the accuracy list
    return accuracy_list
