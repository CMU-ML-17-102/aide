% This compiles all mex files required

mex mexFiles\libsvmread.c -largeArrayDims -outdir mexFiles
mex mexFiles\SDCALocal.cpp -largeArrayDims -outdir mexFiles
mex mexFiles\SVRG.cpp -largeArrayDims -outdir mexFiles
mex mexFiles\SVRGLocal.cpp -largeArrayDims -outdir mexFiles
mex mexFiles\SVRGLocalAcc.cpp -largeArrayDims -outdir mexFiles