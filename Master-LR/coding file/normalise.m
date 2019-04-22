function norm_data = normalise(data)

[rows,columns] = size(data);
norm_data = zeros(rows, columns);

for i = 1:columns
    mean_value = mean(data(:,i));
    std_dev_value = max(data(:,i)) - min(data(:,i));
    
    for j = 1:rows
        norm_data(j,i) = (data(j,i)-mean_value)/std_dev_value;
    end
end
end