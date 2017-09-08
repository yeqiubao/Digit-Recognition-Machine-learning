clear all; close all; clc

A = csvread('train.csv');
B = A(:,2:785);

C = reshape(B(3,:),28,28);
imshow(C)
