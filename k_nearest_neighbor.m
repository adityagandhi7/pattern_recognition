% Reading individual .jpg files and creating Features metrix
a = imgfts('a.jpg');
d = imgfts('d.jpg');
m = imgfts('m.jpg');
n = imgfts('n.jpg');
o = imgfts('o.jpg');
p = imgfts('p.jpg');
q = imgfts('q.jpg');
r = imgfts('r.jpg');
u = imgfts('u.jpg');
w = imgfts('w.jpg');
Features = vertcat(a, d, m, n, o, p, q, r, u, w);

% Populating the Class matrix for each row
ClassA = ones(80,1);
ClassD = repmat(2,80,1);
ClassM = repmat(3,80,1);
ClassN = repmat(4,80,1);
ClassO = repmat(5,81,1);
ClassP = repmat(6,80,1);
ClassQ = repmat(7,79,1);
ClassR = repmat(8,80,1);
ClassU = repmat(9,80,1);
ClassW = repmat(10,80,1);
classes = vertcat(ClassA, ClassD, ClassM, ClassN, ClassO, ClassP, ClassQ, ClassR, ClassU, ClassW);

% Creating the affinity matrix, finding the second nearest neighbor
D=dist2(Features, Features);
[D_sorted, D_index] = sort (D,2);

% Creating matrix that has the predicted class
Predicted_Classes=classes(D_index(:,2));

% Finding the training accuracy
index_check = classes == Predicted_Classes;
correct = sum(index_check);
accuracy = (correct*100)/800;

% Obtaining the test features and creating the true test class
TestFeatures = imgfts('test.jpg');
TrueTestClass=[1*ones(1,7) 2*ones(1,7) 3*ones(1,7) 4*ones(1,7) ...
5*ones(1,7) 6*ones(1,7) 7*ones(1,7) 8*ones(1,7) 9*ones(1,7) 10*ones(1,7)];
load Reorder
TrueTestClass = TrueTestClass (ReorderIndex);
clear ReorderIndex;

% Comparing with the training class
D=dist2(TestFeatures,Features);
[D_sorted,D_index]=sort(D,2);
PredictedTestClass=classes(D_index(:,1));

% Calculate accuracy of the test data
TrueTestClass = TrueTestClass';
index_check_test = TrueTestClass == PredictedTestClass;
correct = sum(index_check_test);
accuracy_test = (correct*100)/70;

% Printing image for classification
showcharlabels ('test.jpg', PredictedTestClass, [1 2 3 4 5 6 7 8 9 10]);

% 1.8 Normalizing the features of the Features Matrix
MeanFeatures = mean(Features);
SdFeatures = std(Features);
NormalizedFeatures = Features - MeanFeatures;
for i = 1:size(NormalizedFeatures,1)
    for j = 1:size(NormalizedFeatures,2)
        NormalizedFeatures(i,j) = NormalizedFeatures(i,j) / SdFeatures(1,j);
    end
end

% 1.8 Calculating training accuracy using normalized features
D=dist2(NormalizedFeatures, NormalizedFeatures);
[D_sorted, D_index] = sort (D,2);
Predicted_Classes_Norm=classes(D_index(:,2));
index_check_norm = classes == Predicted_Classes_Norm;
correct = sum(index_check_norm);
accuracy_norm = (correct*100)/800;

% 1.8 Calculating test accuracy using normalized features
Norm_TestFeatures = TestFeatures - MeanFeatures;
for i = 1:size(Norm_TestFeatures,1)
    for j = 1:size(Norm_TestFeatures,2)
        Norm_TestFeatures(i,j) = Norm_TestFeatures(i,j) / SdFeatures(1,j);
    end
end
D=dist2(Norm_TestFeatures,NormalizedFeatures);
[D_sorted,D_index]=sort(D,2);
Norm_PredictedTestClass=classes(D_index(:,1));
index_check_test_norm = TrueTestClass == Norm_PredictedTestClass;
correct = sum(index_check_test_norm);
accuracy_test_norm = (correct*100)/70;
showcharlabels ('test.jpg', Norm_PredictedTestClass, [1 2 3 4 5 6 7 8 9 10]);

% 1.9 Choosing the 5 nearest neighbors and calculating training accuracy
D=dist2(NormalizedFeatures, NormalizedFeatures);
[D_sorted, D_index] = sort (D,2);
Predicted_Classes_Norm_All=classes(D_index(:,2:6));
Predicted_Classes_Norm_NN5 = mode(Predicted_Classes_Norm_All');
Predicted_Classes_Norm_NN5 = Predicted_Classes_Norm_NN5';

index_check_norm_NN5 = classes == Predicted_Classes_Norm_NN5;
correct = sum(index_check_norm_NN5);
accuracy_norm_NN5 = (correct*100)/800;

% 1.9 Calculating the test accuracy
D=dist2(Norm_TestFeatures, NormalizedFeatures);
[D_sorted, D_index] = sort (D,2);
Predicted_Classes_Norm_All_test=classes(D_index(:,1:5));
Predicted_Classes_Norm_NN5_test = mode(Predicted_Classes_Norm_All_test');
Predicted_Classes_Norm_NN5_test = Predicted_Classes_Norm_NN5_test';

index_check_norm_NN5_test = TrueTestClass == Predicted_Classes_Norm_NN5_test;
correct = sum(index_check_norm_NN5_test);
accuracy_norm_NN5_test = (correct*100)/70;
showcharlabels ('test.jpg', Predicted_Classes_Norm_NN5_test, [1 2 3 4 5 6 7 8 9 10])


% 1.10 Using imgfits2 to calculate the training and test accuracy
% Reading individual .jpg files and creating Features metrix
a2 = imgfts2('a.jpg');
d2 = imgfts2('d.jpg');
m2 = imgfts2('m.jpg');
n2 = imgfts2('n.jpg');
o2 = imgfts2('o.jpg');
p2 = imgfts2('p.jpg');
q2 = imgfts2('q.jpg');
r2 = imgfts2('r.jpg');
u2 = imgfts2('u.jpg');
w2 = imgfts2('w.jpg');
Features2 = vertcat(a2, d2, m2, n2, o2, p2, q2, r2, u2, w2);

MeanFeatures2 = mean(Features2);
SdFeatures2 = std(Features2);
NormalizedFeatures2 = Features2 - MeanFeatures2;
for i = 1:size(NormalizedFeatures2,1)
    for j = 1:size(NormalizedFeatures2,2)
        NormalizedFeatures2(i,j) = NormalizedFeatures2(i,j) / SdFeatures2(1,j);
    end
end

% Creating the affinity matrix, find the 5th nearest neighbor for
% training data
D=dist2(NormalizedFeatures2, NormalizedFeatures2);
[D_sorted, D_index] = sort (D,2);
Predicted_Classes_All2=classes(D_index(:,2:6));
Predicted_Classes2 = mode(Predicted_Classes_All2');
Predicted_Classes2 = Predicted_Classes2';

index_check2 = classes == Predicted_Classes2;
correct = sum(index_check2);
accuracy2 = (correct*100)/800;


% 1.10 Creating the affinity matrix, find the 5th nearest neighbor for
% test data

% Obtaining the test features and creating the true test class
TestFeatures2 = imgfts2('test.jpg');
TrueTestClass2=[1*ones(1,7) 2*ones(1,7) 3*ones(1,7) 4*ones(1,7) ...
5*ones(1,7) 6*ones(1,7) 7*ones(1,7) 8*ones(1,7) 9*ones(1,7) 10*ones(1,7)];
load Reorder
TrueTestClass2 = TrueTestClass2 (ReorderIndex);
clear ReorderIndex;
TrueTestClass2 = TrueTestClass2';

Norm_Test_Features2 = TestFeatures2 - MeanFeatures2;
for i = 1:size(Norm_Test_Features2,1)
    for j = 1:size(Norm_Test_Features2,2)
        Norm_Test_Features2(i,j) = Norm_Test_Features2(i,j) / SdFeatures2(1,j);
    end
end

D=dist2(Norm_Test_Features2, NormalizedFeatures2);
[D_sorted, D_index] = sort (D,2);
Predicted_Classes_All_test_2=classes(D_index(:,1:5));
Predicted_Classes_test2 = mode(Predicted_Classes_All_test_2');
Predicted_Classes_test2 = Predicted_Classes_test2';

index_check_test2 = TrueTestClass2 == Predicted_Classes_test2;
correct = sum(index_check_test2);
accuracy_test2 = (correct*100)/70;
showcharlabels ('test.jpg', Predicted_Classes_test2, [1 2 3 4 5 6 7 8 9 10])
