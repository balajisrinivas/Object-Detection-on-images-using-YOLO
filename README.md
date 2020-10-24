# Object Detection on images using YOLO

**YOLO (You Only Look Once)** is a very powerful and a fast algorithm in object detection. A strong understanding of the algorithm is essential before we start to code.

**Some important papers to start with -**

There are three papers you need to go through *(Maybe difficult to understand initially, but worth reading it)*

- [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo.pdf)
- [YOLO9000: Better, Faster, Stronger](https://pjreddie.com/media/files/papers/YOLO9000.pdf)
- [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

We are going to use YOLO v3 for our coding purpose in this repository.

*Before going to code, we need to download some important YOLO files. It's the folder that's present in this repository as yolo-coco*

The three files that needs to be downloaded are -

- [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
- [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

Download these files and save it inside a folder. Name the folder anything you wish, but I have named it as **yolo-coco** just because of the fact that we are going to use the coco dataset objects.

Create a folder **images** and have some pictures inside it to test the object detection.

The **yolo.py** has the script to detect the objects in the images.

Make sure you have numpy and opencv installed. If not install them using pip

```
pip install numpy
pip install opencv-python
```

I am using the ***numpy*** version ***1.17.4*** and ***opencv*** version ***3.4.2***

You can now run the file by giving this command on your command promt

```
python yolo.py --image images/ipl.jpeg
```

You can use any image you want after the `--image` argument. Make sure you give the right path.

Press **q** to quit the window of the image showing object detection

Feel free to star ⭐ this repo if you find this useful and visit my [YouTube](https://www.youtube.com/channel/UCaOiKrS-R1Nuya0pxjQ68fA) channel if you have time.

Thanks!

