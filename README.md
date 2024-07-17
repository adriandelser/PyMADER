# PyMADER
A python implementation of <a href="https://github.com/mit-acl/mader" target="_blank">MADER</a>, a multi agent drone deconfliction algorithm from MIT. 

## Some Visualisations of the algorithm in action:
<!-- <p align="center">
  <img src="assets/demo.gif" width="45%" />
  <img src="assets/demo1.png" width="45%" />
</p> -->

<div style="display: flex; justify-content: center; align-items: center;">
  <div style="text-align: center; margin-right: 10px;">
    <img src="assets/demo.gif" width="790px" />
    <div>Agent avoiding two dynamic obstacles with separating planes visible</div>
  </div>
  <div style="text-align: center; margin-left: 10px;">
    <img src="assets/demo1.png" width="900px" />
    <div>Snapshot showing accepted (green), rejected (red) and unexplored (coloured by relative cost heuristic) control points during the Octopus search</div>
  </div>
</div>