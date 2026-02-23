[!] High priority :

- [ ] Remove draw-card in the rust simulator as a manual action as it should be automatic. Currently there is 5 actions on average during the DRL training, cutting that by 1 means 20% less steps per turn.

- [ ] Encode attached tool description (text embedding) on played Pokemon cards in observation tensor (`src/rl/observation.rs`). Currently tools only get a 1-bit `is_tool` flag but the model has no way to know *what* tool is attached to a Pokemon.

Medium priority :


- [ ] Design and Implement AI deckbuilder

Low priority :

- [ ] Explore others learning algorithms apart from PPO