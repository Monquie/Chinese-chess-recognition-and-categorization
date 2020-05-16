[m,n]=size(error2World);
sumx=0;
sumy=0;
for i = 1:1:m
    sumx = sumx + error2World(i,1);
end
sumx = sumx/m;
for i = 1:1:m
    sumy = sumy +error2World(i,2);
end
sumy = sumy/m;