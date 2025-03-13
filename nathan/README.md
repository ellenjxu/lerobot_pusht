`python -m nathan.mask --dataset "ellen2imagine/pusht_green1" --episode 0 --sequence --start 0 --num-frames 10 --skip 5 --output "./output/t_detection_sequence"`

`python -m nathan.get_t_info.color_picker` is for selecting the color of the T shape to be used in the mask.py script.

`python -m nathan.get_t_info.t_position_ui` is for selecting the position of the T shape to be used in the mask.py script for getting IoUs

`python -m nathan.get_t_info.mask --dataset "ellen2imagine/pusht_green1" --episode 0 --sequence --start 0 --num-frames 10 --skip 5 --output "./output/t_detection_sequence"` is for detecting the T shape in the video.

