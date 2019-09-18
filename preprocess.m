function out= preprocess(GT,Q,features_matrix,videosToIndex,list)
out = cell(size(Q));
for query = 1: length(Q),
    
    out{query} = cell(3,length(videosToIndex));
    q = Q{query};
   
    %% making positives for this query
    videos = videosToIndex;
    stack = [];
    for concept=1:length(q),
        a = q(concept);
        
        inds = find(GT{a}(:,3)==1);
        
        videos = intersect(videos,unique(GT{a}(inds,1)));
        
    end
    stack = list(ismember(list(:,1),videos),:);
    
    %%purify this stack
    %        StackBadInds = [];
    %        for i=1:length(stack),
    %           for concept=1:length(q),
    %             a = q(concept);
    %             indexes = find(find(GT{a}(:,3)==-1));
    %             indexes = intersect(indexes,find(GT{a}(:,1)==stack(i,1)));
    %             indexes = intersect(indexes,find(GT{a}(:,2)==stack(i,2)));
    %             if(~isempty(indexes)),
    %                StackBadInds = [StackBadInds,i];
    %             end
    %           end
    %        end
    %        stack(StackBadInds,:) = [];
    videos = unique(stack(:,1));
    for v=1:length(videos),
        inds = find(stack(:,1)==videos(v));
        vidind = find(videosToIndex==videos(v));
        out{query}{1,vidind} = pairToIndex(stack(inds,:),list);
    end
    
    
    
    
    %% making negative for this query
    videos = videosToIndex;
    stack = [];
    for concept=1:length(q),
        a = q(concept);
        inds = find(GT{a}(:,3)==-1);
        videos = intersect(videos,unique(GT{a}(inds,1)));
    end
    stack = list(ismember(list(:,1),videos),:);
    
    %%purify this stack (videos which have a positive for one of concepts should be excluded)
    StackBadVids = [];
    NegVids = unique(stack(:,1));
    for concept=1:length(q),
        a=q(concept);
        for i=1:length(NegVids),
            inds = find(GT{a}(:,3)==1);
            if(ismember(NegVids(i),unique(GT{a}(inds,1)))),
                StackBadVids = [StackBadVids, NegVids(i)];
            end
        end
    end
    stack(ismember(stack(:,1),StackBadVids),:) = [];
    %%purify this stack (shots which doesn't have anytag for all concepts should be removed)
    
    badNeginds = [];
    
    for i=1:size(stack,1),
        
        isbad = 0;
       for concept = 1:length(q),
           a = q(concept);
           inds = find(GT{a}(:,1)==stack(i,1));
           inds = intersect(inds,find(GT{a}(:,2)==stack(i,2)));
           if(isempty(inds)),
              isbad = 1;
           else
               isbad = 0;
               continue;
           end
       end
       if(isbad),
          badNeginds = [badNeginds;i]; 
       end
        
    end
    stack(badNeginds,:) = [];
    
    videos = unique(stack(:,1));
    for v=1:length(videos),
        inds = find(stack(:,1)==videos(v));
        vidind = find(videosToIndex==videos(v));
        out{query}{3,vidind} = pairToIndex(stack(inds,:),list);
    end
    
    %% makin middle class for this query
    videos = [];
    stack = [];
    for concept=1:length(q),
        a = q(concept);
        
        inds = find(GT{a}(:,3)==1);
        
        videos = union(videos,unique(GT{a}(inds,1)));
        
    end
    stack = list(ismember(list(:,1),videos),:);
    
    videos = unique(stack(:,1));
    for v=1:length(videos),
        inds = find(stack(:,1)==videos(v));
        vidind = find(videosToIndex==videos(v));
        out{query}{2,vidind} = pairToIndex(stack(inds,:),list);
    end
    %%videos which have all the concepts as postivies are in positive class
    %%here we excludethose from the middle class
    for v=1:length(videosToIndex),
       if(~isempty(out{query}{2,v})&& ~isempty(out{query}{1,v})),
           out{query}{2,v} = [];
       end
        
    end
     
end


end