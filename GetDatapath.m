function datapath = GetDatapath()
  
  if strcmp(pwd, 'C:\!EWork\!TexFiles\dist_svrg\AIDE_experiments')
    datapath = 'C:\Data\'; 
  elseif strcmp(pwd, 'F:\!Ework\!TexFiles\dist_svrg\AIDE_experiments')
    datapath = 'F:\!Ework\Data\';
  elseif strcmp(pwd, 'D:\!EWork\!TexFiles\dist_svrg\AIDE_experiments')
    datapath = 'D:\Data\';
  end
  
end