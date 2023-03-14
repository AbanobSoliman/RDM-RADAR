The following steps are used to implement 2D-CFAR in MATLAB:

1- Determine the number of Training cells for each dimension Tr and Td. Similarly, pick the number of guard cells Gr and Gd.
2- Slide the Cell Under Test (CUT) across the complete cell matrix
3- Select the grid that includes the training, guard and test cells. Grid Size = (2Tr+2Gr+1)(2Td+2Gd+1).
4- The total number of cells in the guard region and cell under test. (2Gr+1)(2Gd+1).
5- This gives the Training Cells : (2Tr+2Gr+1)(2Td+2Gd+1) - (2Gr+1)(2Gd+1)
6- Measure and average the noise across all the training cells. This gives the threshold
7- Add the offset (if in signal strength in dB) to the threshold to keep the false alarm to the minimum.
8- Determine the signal level at the Cell Under Test.
9- If the CUT signal level is less than the Threshold, assign a value of 0, else equate it to 1.
10- Since the cell under test are not located at the edges, due to the training cells occupying the edges, we suppress the edges to zero. Any cell value that is neither 1 nor a 0, assign it a zero.
11- Based on the i,j limits, I used the union function to select the outlier regions on bith range-doppler axes to be set with zeros.
