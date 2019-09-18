function out = TrainSampleBuilder(indexSet,TrainSet,videosToIndex)
    out = cell(1,length(indexSet));
    for i=1:length(indexSet),
       for j=1:size(indexSet{i},1),
            out{i}{j} = [];
           for k=1:size(indexSet{i},2),
              if(~isempty(indexSet{i}{j,k})),
                  if(~ismember(videosToIndex(k),TrainSet)),
                      continue;
                  end
                 out{i}{j} = [out{i}{j},k]; 
                 
              end
          end
       end
    end


end