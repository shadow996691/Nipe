#include <iostream>
#include <vector>
#include <cmath>
#include "raylib.h"
#include <torch/torch.h>
#include <vector>
#include <fstream>
using namespace std;


struct RLNetwork : torch::nn::Module {
    torch::nn::ModuleList layers{nullptr};

    // Create network from YOUR parameters
    // inputSize = size of inputs[], outputSize = size of outputs[]
    // hiddenLayers = count, hiddenNeurons[] = neurons per layer
    RLNetwork(int inputSize, int outputSize, int hiddenLayers, int hiddenNeurons[]) {
        layers = register_module("layers", torch::nn::ModuleList());

        int prevSize = inputSize;

        // 1. Create hidden layers
        for(int i = 0; i < hiddenLayers; i++){
            int currSize = hiddenNeurons[i];
            layers->push_back(torch::nn::Linear(prevSize, currSize));
            prevSize = currSize;
        }

        // 2. Create output layer
        layers->push_back(torch::nn::Linear(prevSize, outputSize));
    }
    
    torch::Tensor forward(torch::Tensor x){
	    // Hidden layers with ReLU
	    for(size_t i = 0; i < layers->size() - 1; i++){
		    x = torch::relu(layers[i]->as<torch::nn::Linear>()->forward(x));
	    }
	    // Last layer - no ReLU, raw Q-values
	    x = layers[layers->size() - 1]->as<torch::nn::Linear>()->forward(x);
	    return x;
    }
};

// Helper - creates network in 1 line
std::shared_ptr<RLNetwork> CreateNetwork(int inputSize, int outputSize, int hiddenLayers, int hiddenNeurons[]){
    auto net = std::make_shared<RLNetwork>(inputSize, outputSize, hiddenLayers, hiddenNeurons);
    return net;
}

void DrawClickCoordinates2D(
int gridSpacing = 50,
Color markerColor = RED,
Color textColor = DARKGRAY,
int fontSize = 14,
float markerRadius = 4.0f,
bool connectPoints = false,
bool showGrid = true,
MouseButton addBtn = MOUSE_BUTTON_LEFT,
MouseButton clearBtn = MOUSE_BUTTON_RIGHT)
{
static std::vector<Vector2> points;
if (showGrid && gridSpacing > 0)
{
for (int x = 0; x < GetScreenWidth(); x += gridSpacing)
DrawLine(x, 0, x, GetScreenHeight(), Fade(LIGHTGRAY, 0.35f));
for (int y = 0; y < GetScreenHeight(); y += gridSpacing)
DrawLine(0, y, GetScreenWidth(), y, Fade(LIGHTGRAY, 0.35f));
}
if (IsMouseButtonPressed(addBtn)) points.push_back(GetMousePosition());
if (IsMouseButtonPressed(clearBtn)) points.clear();
for (size_t i = 0; i < points.size(); ++i)
{
if (connectPoints && i > 0) DrawLineV(points[i - 1], points[i], Fade(markerColor, 0.6f));
DrawCircleV(points[i], markerRadius, markerColor);
DrawText(TextFormat("(%d, %d)", (int)points[i].x, (int)points[i].y),
(int)points[i].x + 8, (int)points[i].y - 8, fontSize, textColor);
}
}

Color GetMapColor(Image img, float x, float y){
int px = (int)x;
int py = (int)y;
if(px < 0 || py < 0 || px >= img.width || py >= img.height) return RED;
return GetImageColor(img, px, py);
}

bool IsRed(Color c){ return c.r > 200 && c.g < 80 && c.b < 80; }
bool IsBlue(Color c){ return c.b > 200 && c.r < 80 && c.g < 80; }
// YOUR track: Dark green = SPAWN, Bright green = FINISH
bool IsSpawnGreen(Color c){ return c.g > 100 && c.g < 220 && c.r < 80 && c.b < 80; } // dark green square
bool IsFinishGreen(Color c){ return c.g > 240 && c.r < 80 && c.b < 80; } // bright green line

struct RayHit {
    float dist;
    Vector2 end;
    bool hit;
};

RayHit CastRay(Image img, float x, float y, float angleDeg, float maxDist){
    RayHit result = {maxDist, {x, y}, false};
    float rad = angleDeg * DEG2RAD;

    for(float d = 0; d < maxDist; d += 3){
        float px = x + cos(rad) * d;
        float py = y + sin(rad) * d;
        if(IsRed(GetMapColor(img, px, py))){
            result.dist = d;
            result.end = {px, py};
            result.hit = true;
            return result;
        }
    }
    result.end = {x + cos(rad) * maxDist, y + sin(rad) * maxDist};
    return result;
}

Vector2 FindSpawn(Image img){
    for(int y = 0; y < img.height; y++){
        for(int x = 0; x < img.width; x++){
            Color c = GetImageColor(img, x, y);
            if(IsSpawnGreen(c)){
                return {(float)x, (float)y};
            }
        }
    }
    return {50, 1350}; // fallback if no green found
}

int main(){
	string race;
	cout << "choose your track\n";
	cin >> race;
	InitWindow(400,400,"testing");
	SetTargetFPS(50);
	Camera2D camera = {0};
	camera.offset = (Vector2){200, 200};
	camera.zoom = 0.5f;
	Image map;
	map = LoadImage(race.c_str());
	Texture2D texture = LoadTextureFromImage(map);
	Vector2 spawn = FindSpawn(map);
	Vector2 checkpoint = spawn; // respawn point, updates on BLUE
	bool hasCheckpoint = false;
	bool hasWon = false;
	
	float carX = spawn.x;
	float carY = spawn.y;
	float carAngle = 135;
	float carSpeed = 0;
	// --- TRAINING SETUP ---
	int hidden[] = {64, 64};
	auto net = CreateNetwork(5, 4, 2, hidden);
	torch::optim::Adam optimizer(net->parameters(), 0.001);
	float epsilon = 1.0f; // 1.0 = 100% random, will decay
	
	fstream main;
	main.open("model.pt",ios::in);
	if(main.is_open()){
		main.close();
		torch::load(net,"model.pt");
		cout << "model has been loaded successfully.\n";
		epsilon = 0.0f; 
	}
	
	
	float epsilonMin = 0.05f;
	float epsilonDecay = 0.9995f;
	float gamma = 0.99f; // how much it cares about future
	int episode = 0;
	while(!WindowShouldClose()){
		// --- 1. CURRENT STATE ---
		RayHit left  = CastRay(map,carX,carY,carAngle -60,200);
		RayHit midL  = CastRay(map,carX,carY,carAngle -20,200);
		RayHit midR  = CastRay(map,carX,carY,carAngle +20,200);
		RayHit right = CastRay(map,carX,carY,carAngle +60,200);
		
		torch::Tensor state = torch::tensor({left.dist/200.0f, midL.dist/200.0f, midR.dist/200.0f, right.dist/200.0f, carSpeed/5.0f});
		
		// --- 2. AI DECIDES (epsilon-greedy) ---
		int action;
		if ((float)rand()/RAND_MAX < epsilon) {
		    action = rand() % 4; // explore: random
		} else {
		    torch::NoGradGuard no_grad;
		    action = net->forward(state).argmax().item<int>(); // exploit: best
		}
		
		bool turnLeft  = (action == 0);
		bool turnRight = (action == 1);
		bool gas       = (action == 2);
		bool brake     = (action == 3);
		
		if(turnLeft)  carAngle -= 3.0f;
		if(turnRight) carAngle += 3.0f;
		if(gas) carSpeed += 0.2f;
		else if(brake) carSpeed -= 0.2f;
		else carSpeed *= 0.97f;
		if(carSpeed > 5) carSpeed = 5;
		if(carSpeed < -2) carSpeed = -2;
		
		carX += cos(carAngle * DEG2RAD) * carSpeed;
		carY += sin(carAngle * DEG2RAD) * carSpeed;
		
		
		// --- 3. REWARD + LEARN (SMARTER) ---
		float reward = 0.0f;
		bool done = false;
		
		Color underCar = GetMapColor(map, carX, carY);
		
		// 1. TIME PENALTY - punishes sitting still / going slow
		// If it just sits there, it bleeds points
		reward -= 0.05f;
		
		// 2. SPEED REWARD - but punish going backwards
		if (carSpeed > 0) {
		    reward += carSpeed * 0.05f; // fast = good
		} else {
		    reward -= 0.5f; // backwards = bad, it will learn to not reverse
		}
		
		// 3. WALL PROXIMITY PUNISHMENT - THIS IS THE KEY
		// Use the rays we already have
		float minDist = left.dist;
		if (midL.dist < minDist) minDist = midL.dist;
		if (midR.dist < minDist) minDist = midR.dist;
		if (right.dist < minDist) minDist = right.dist;
		
		if (minDist < 20) reward -= 2.0f;      // VERY close = big punishment
		else if (minDist < 40) reward -= 0.5f; // close = small punishment
		else if (minDist > 80) reward += 0.2f; // in middle = small reward
		
		// 4. SPINNING PUNISHMENT
		// If it just spins in circles, punish it
		if (turnLeft && turnRight) reward -= 1.0f; // can't happen now, but keep
		if ((action == 0 || action == 1) && carSpeed < 0.5f) {
		    reward -= 0.3f; // turning while stopped = stupid
		}
		if(IsMouseButtonPressed(MOUSE_BUTTON_LEFT)){
			torch::save(net,"model.pt");
			cout << "model has been saved successfully.\n";
		}
		if(IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)){
			cout << "change the racetrack:\n";
			cin >> race;
			torch::load(net,"model.pt");
			UnloadImage(map);
			UnloadTexture(texture);
			map = LoadImage(race.c_str());
			texture = LoadTextureFromImage(map);
			spawn = FindSpawn(map);
			carX = spawn.x;
			carY = spawn.y;
			carSpeed = 0;
			carAngle = 135;
			hasCheckpoint = false;
			hasWon = false;
		}
		
		// 5. CRASH / CHECKPOINT / WIN
		if(IsRed(underCar)){
		    reward = -20.0f; // was -10, now -20, much harsher
		    done = true;
		    carX = checkpoint.x;
		    carY = checkpoint.y;
		    carSpeed = 0;
		}
		else if(IsBlue(underCar)){
		    if(!hasCheckpoint) reward += 20.0f; // was 10, now 20
		    checkpoint = {carX, carY};
		    hasCheckpoint = true;
		}
		else if(IsFinishGreen(underCar)){
		    if(hasCheckpoint && !hasWon) reward += 200.0f; // was 100
		    if(hasCheckpoint) hasWon = true;
		}
		// --- NEXT STATE (you deleted this) ---
		RayHit nLeft  = CastRay(map,carX,carY,carAngle -60,200);
		RayHit nMidL  = CastRay(map,carX,carY,carAngle -20,200);
		RayHit nMidR  = CastRay(map,carX,carY,carAngle +20,200);
		RayHit nRight = CastRay(map,carX,carY,carAngle +60,200);
		torch::Tensor nextState = torch::tensor({nLeft.dist/200.0f, nMidL.dist/200.0f, nMidR.dist/200.0f, nRight.dist/200.0f, carSpeed/5.0f});
		
		// --- Q-LEARNING UPDATE ---
		optimizer.zero_grad();
		torch::Tensor q_values = net->forward(state);
		torch::Tensor q_next;
		{
		    torch::NoGradGuard no_grad;
		    q_next = net->forward(nextState);
		}
		float maxQ = q_next.max().item<float>();
		float target = reward + (done ? 0 : gamma * maxQ);
		
		torch::Tensor targetTensor = q_values.clone().detach();
		targetTensor[action] = target;
		
		auto loss = torch::mse_loss(q_values, targetTensor);
		loss.backward();
		optimizer.step();
		
		// Decay exploration
		if(epsilon > epsilonMin) epsilon *= epsilonDecay;
		if(done) episode++;
		BeginDrawing();
		ClearBackground(BLACK);
		camera.target = (Vector2){carX, carY};
		BeginMode2D(camera);
		DrawTexture(texture, 0, 0, WHITE);
		// --- Car drawing vars (were deleted) ---
		Rectangle carRect = {carX, carY, 30, 15};
		Vector2 carOrigin = {15, 7.5f};
		float carRotation = carAngle;
		Vector2 carPos = {carX,carY};
		DrawRectanglePro(carRect, carOrigin, carRotation, YELLOW);
		DrawLineV(carPos,left.end,GREEN);
		DrawLineV(carPos,midL.end,GREEN);
		DrawLineV(carPos,midR.end,GREEN);
		DrawLineV(carPos,right.end,GREEN);
		EndMode2D();
		if(hasCheckpoint && !hasWon) DrawText("CHECKPOINT!", 10, 10, 20, BLUE);
		if(hasWon) DrawText("YOU WIN!", 120, 180, 40, GREEN);
		if(IsKeyDown(KEY_LEFT_CONTROL))DrawText(TextFormat("Ep: %d Eps: %.2f Loss: %.3f", episode, epsilon, loss.item<float>()), 10, 30, 20, WHITE);
		EndDrawing();
	}
	UnloadTexture(texture);
	UnloadImage(map);
	CloseWindow();
}
