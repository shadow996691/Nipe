intro : 🌎 

an open-source RL Ai neural network. ready to train anytime.
made of C++. simple project. 
requirements : 🔧 
LibTorch,
Raylib,
Clang++ and GCC compilers.


USAGE : 📖

compile the main.cpp using this command.
```
g++ -std=c++17 -o main main.cpp -I/root/nipe/raylib/include -I/root/nipe/libtorch/include -I/root/nipe/libtorch/include/torch/csrc/api/include /root/nipe/raylib/lib/libraylib.a -L/root/nipe/libtorch/lib -ltorch -ltorch_cpu -lc10 -lGL -lm -lpthread -ldl -lrt -lX11 -Wl,-rpath,/root/nipe/libtorch/lib
```
and run.
```
./main
```
the track might looks ugly. but you can add your own custom track.
by using the regular colors. like for example red for traps/walls, bright green for goal/finish line, Dark green for spawn/start, blue for checkpoints. put them into PNG image and 128×128 pixels


Accessories : ✂️ 

you can also check the episodes and loss by hodling CTRL.
and if you want to save the model click left mouse button. and if you  want to change the track during the training. Just click Right mouse button and type your track name and with his extention for example like:
```
track2.png
```
in the terminal.


need a help ?.
join my discord DM link.
```
https://discord.gg/ua7pQqFK
```


hope you satisfied 😊.
