function out= pairToIndex(input,list),
    out = zeros(size(input,1),1);
   
    for i=1:size(input,1),
       vidId = input(i,1);
       shotId = input(i,2);
       listId = intersect(find(list(:,1)==vidId),find(list(:,2)==shotId));
        if(length(listId)~=1),
          keyboard; 
       end
       out(i) = listId;
      
        
    end
    

end