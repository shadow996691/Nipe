#include <iostream>
#include <vector>
#include <cmath>
#include "raylib.h"
#include "physics.hpp" // your physics
#include <torch/torch.h>

Physics physics;

// ==================== RAYCAST (INT) ====================
struct RaycastHit { int distance = 9999; bool hit = false; };
struct Raycaster {
    RaycastHit down, up, left, right;
    int GetDown() { return down.distance; }
    int GetUp() { return up.distance; }
    int GetLeft() { return left.distance; }
    int GetRight() { return right.distance; }

    RaycastHit CastRay(Vector2 origin, Vector2 dir, float maxDist) {
        RaycastHit r; r.distance = (int)maxDist;
        float len = sqrtf(dir.x*dir.x + dir.y*dir.y);
        if(len==0) return r; dir.x/=len; dir.y/=len;
        for(auto &b : physics.bodies) {
            if(b.layer != LAYER_GROUND) continue;
            for(int s=0; s<(int)maxDist; s+=5) {
                Vector2 p = {origin.x + dir.x*s, origin.y + dir.y*s};
                if(CheckCollisionPointRec(p, b.rect)) { r.distance=s; r.hit=true; return r; }
            }
        }
        return r;
    }
    void Update(Vector2 pos) {
        Vector2 c = {pos.x+15, pos.y+25};
        down = CastRay(c, {0,1}, 200);
        up = CastRay(c, {0,-1}, 200);
        left = CastRay(c, {-1,0}, 200);
        right = CastRay(c, {1,0}, 200);
    }
    void Draw(Vector2 pos) {
        Vector2 c = {pos.x+15, pos.y+25};
        DrawLineEx(c, {c.x, c.y+down.distance}, 2, down.hit?RED:GRAY);
        DrawLineEx(c, {c.x, c.y-up.distance}, 2, up.hit?RED:GRAY);
        DrawLineEx(c, {c.x-left.distance, c.y}, 2, left.hit?RED:GRAY);
        DrawLineEx(c, {c.x+right.distance, c.y}, 2, right.hit?RED:GRAY);
    }
} g_raycast;

// ==================== BALANCE INPUT ====================
struct BalanceInput {
    float torsoLean=0, legL=0, legR=0, armL=0, armR=0;
    int posX=0, posY=0, finishX=1100, finishY=400, distanceToFinish=0;
    int rayGround=9999, rayCeiling=9999, rayWallLeft=9999, rayWallRight=9999;
    bool isAtFinish=false;
    void SetTorso(float v){torsoLean=v;} void SetLegL(float v){legL=v;}
    void SetLegR(float v){legR=v;} void SetArmL(float v){armL=v;} void SetArmR(float v){armR=v;}
} g_input;

// ==================== NEURAL NETWORK (RL) ====================
struct PolicyNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    PolicyNet() {
        fc1 = register_module("fc1", torch::nn::Linear(8, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 5));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::tanh(fc3->forward(x)); // -1 to 1 perfect for your SetXXX
        return x;
    }
};

// ==================== PLAYER / TRACK ====================
struct Player {
    int torso, head, armL, armR, legL, legR, jHead, jArmL, jArmR, jLegL, jLegR;
    void Create(Vector2 pos) {
        torso=physics.Add({pos.x,pos.y,30,50},false,LAYER_PLAYER,LAYER_ALL);
        head=physics.Add({pos.x+5,pos.y-22,20,20},false,LAYER_PLAYER,LAYER_ALL);
        armL=physics.Add({pos.x-16,pos.y+5,12,30},false,LAYER_PLAYER,LAYER_ALL);
        armR=physics.Add({pos.x+34,pos.y+5,12,30},false,LAYER_PLAYER,LAYER_ALL);
        legL=physics.Add({pos.x+2,pos.y+50,14,35},false,LAYER_PLAYER,LAYER_ALL);
        legR=physics.Add({pos.x+16,pos.y+50,14,35},false,LAYER_PLAYER,LAYER_ALL);
        jHead=physics.AddRevoluteJoint(torso,head); jArmL=physics.AddRevoluteJoint(torso,armL);
        jArmR=physics.AddRevoluteJoint(torso,armR); jLegL=physics.AddRevoluteJoint(torso,legL);
        jLegR=physics.AddRevoluteJoint(torso,legR);
    }
    void Update(float dt){
        float p=400.0f;
        physics.bodies[torso].vx+=g_input.torsoLean*p*dt;
        physics.bodies[legL].vx+=g_input.legL*p*dt; physics.bodies[legR].vx+=g_input.legR*p*dt;
        physics.bodies[armL].vx+=g_input.armL*p*dt; physics.bodies[armR].vx+=g_input.armR*p*dt;
    }
    Vector2 GetPos(){return {physics.bodies[torso].rect.x, physics.bodies[torso].rect.y};}
    Vector2 GetCenter(){return {physics.bodies[torso].rect.x+15, physics.bodies[torso].rect.y+25};}
    void Draw(){
        DrawRectangleRec(physics.bodies[torso].rect,BLUE);
        DrawRectangleRec(physics.bodies[head].rect,BEIGE);
        DrawRectangleRec(physics.bodies[armL].rect,DARKBLUE);
        DrawRectangleRec(physics.bodies[armR].rect,DARKBLUE);
        DrawRectangleRec(physics.bodies[legL].rect,SKYBLUE);
        DrawRectangleRec(physics.bodies[legR].rect,SKYBLUE);
    }
};

struct Track {
    Rectangle finishLine={1100,400,20,200};
    std::vector<int> ground;
    void Create(){ ground.push_back(physics.Add({0,600,1280,40},true,LAYER_GROUND,LAYER_ALL)); }
    void Draw(){ for(int id:ground) DrawRectangleRec(physics.bodies[id].rect,DARKGRAY); DrawRectangleRec(finishLine,GREEN); }
    Vector2 GetFinishPos(){return {finishLine.x, finishLine.y};}
};

// ==================== OBSERVATION + REWARD ====================
torch::Tensor GetObservation(Player &player){
    float px = player.GetPos().x / 1280.0f;
    float py = player.GetPos().y / 720.0f;
    float dist = (float)g_input.distanceToFinish / 1280.0f;
    float rg = (float)g_raycast.GetDown() / 200.0f;
    float ru = (float)g_raycast.GetUp() / 200.0f;
    float rl = (float)g_raycast.GetLeft() / 200.0f;
    float rr = (float)g_raycast.GetRight() / 200.0f;
    float vx = physics.bodies[player.torso].vx / 500.0f;
    return torch::tensor({px, py, dist, rg, ru, rl, rr, vx});
}

int main(){
    InitWindow(1280,720,"Raylib + LibTorch RL - Training");
    SetTargetFPS(60);
    physics.gravity=800.0f;

    auto policy = std::make_shared<PolicyNet>();
    torch::optim::Adam optimizer(policy->parameters(), 0.001);

    Player player; Track track;
    track.Create();
    player.Create({100,500});

    int episode=0; float episodeReward=0;
    std::vector<torch::Tensor> saved_log_probs;
    std::vector<float> saved_rewards;

    while(!WindowShouldClose()){
        float dt=GetFrameTime();

        // ===== 1. AI THINKS (INPUT -> OUTPUT) =====
        auto obs = GetObservation(player); // INPUT = X,Y,raycasts
        auto action = policy->forward(obs); // OUTPUT = 5 actions

        // Exploration noise
        auto noise = torch::randn({5}) * 0.2f;
        action = action + noise;
        action = torch::clamp(action, -1, 1);

        // Apply to game
        g_input.SetTorso(action[0].item<float>());
        g_input.SetLegL(action[1].item<float>());
        g_input.SetLegR(action[2].item<float>());
        g_input.SetArmL(action[3].item<float>());
        g_input.SetArmR(action[4].item<float>());

        // ===== 2. GAME UPDATES =====
        player.Update(dt);
        physics.Update(dt);
        g_raycast.Update(player.GetPos());
        
        g_input.posX=(int)player.GetPos().x;
        g_input.posY=(int)player.GetPos().y;
        int dx=(int)track.GetFinishPos().x - g_input.posX;
        int dy=(int)track.GetFinishPos().y - g_input.posY;
        g_input.distanceToFinish=(int)sqrtf((float)(dx*dx+dy*dy));
        g_input.rayGround=g_raycast.GetDown();
        g_input.rayWallRight=g_raycast.GetRight();

        // ===== 3. REWARD =====
        bool fell = physics.check_outside(player.torso, {0,0,1280,720});
        bool finished = CheckCollisionPointRec(player.GetPos(), track.finishLine);
        float reward = 0;
        if(fell) reward=-100;
        else if(finished) reward=+100;
        else {
            reward = -0.01f * g_input.distanceToFinish; // closer is better
            reward += (g_raycast.GetDown() < 40 ? 0.5f : -0.5f); // stay near ground
        }
        episodeReward+=reward;
        saved_rewards.push_back(reward);

        // Save for training (log prob simplified)
        saved_log_probs.push_back(action.mean());

        // ===== 4. DRAW (so you SEE learning) =====
        BeginDrawing();
        ClearBackground(RAYWHITE);
        track.Draw(); player.Draw(); g_raycast.Draw(player.GetPos());
        DrawText(TextFormat("Episode: %d Reward: %.1f",episode,episodeReward),10,10,20,BLACK);
        DrawText(TextFormat("X:%d Y:%d RayGround:%d Dist:%d",g_input.posX,g_input.posY,g_input.rayGround,g_input.distanceToFinish),10,35,18,BLUE);
        DrawText("Training... Raylib + LibTorch together!",10,60,18,DARKGRAY);
        EndDrawing();

        // ===== 5. END OF EPISODE -> TRAIN =====
        if(fell || finished || saved_rewards.size()>500){
            // Compute discounted rewards
            float R=0; std::vector<float> returns;
            for(int i=saved_rewards.size()-1;i>=0;i--){ R=saved_rewards[i]+R*0.99f; returns.insert(returns.begin(),R); }
            
            auto returns_t = torch::tensor(returns);
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std()+1e-6); // normalize

            auto log_probs_t = torch::stack(saved_log_probs);
            auto loss = -(log_probs_t * returns_t).mean(); // INPUT was obs, OUTPUT was action, TARGET is returns

            optimizer.zero_grad();
            loss.backward(); // THIS IS WHERE INPUT/OUTPUT GETS DURING TRAINING
            optimizer.step();

            std::cout << "Episode " << episode << " Loss: " << loss.item<float>() << " Reward: " << episodeReward << std::endl;

            // reset
            physics = Physics(); // reset all bodies
            physics.gravity = 800.0f;
            track.Create(); player.Create({100,500});
            saved_log_probs.clear(); saved_rewards.clear();
            episodeReward=0; episode++;
        }
    }
    torch::save(policy, "policy.pt");
    CloseWindow();
    return 0;
}