Voice Activity Detection Optimization & Deployment 
==================================================

VAD Optimization
----------------

The first step we took optimize `is_silence` was to reduce the delay.
We noticed that one of the most time costly operations is downsamplig, by setting `downsampling_rate` avoiding it we reduce the delay
from $32\mu s$ to $6\mu s$.
Also `frame_length_in_s` has some influence on time because it influences the number and size of the FFTs needed to performed.
If we use `frame_length_in_s=.001` delay is halved to $3\mu s$ but now `is_silence` now only outputs `0` and the accuracy drops to 88%.
On the other hand, duration of words is of the order of tenths of second, so we set `frame_length_s=.1` and boost accuracy to 96% keeping
the delay to $6\mu s$.
We have then compared energy in silence and noisy files and figured that `dbFSthresh` should be in $[-110, -130]$.
Moreover we made the sendible assumption that a silent file is at least $75%$ silent, so we looked for the value of `duration_thres` in  $[.75, 1]$.
We performed a grid search over `dbFSthresh` and `duration_thres` on the specified intervals respectively with steps $1$ and $.05$. We found that
the best configuration is `dbFSthresh=-121` and `duration_thresh=.85` and achieves $99%$ accuracy

