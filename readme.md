## Project Description

This project implements an end-to-end **rating-based recommender system** on the **MovieLens 32M** dataset using PyTorch. The goal is to learn user and item (movie) representations that predict explicit ratings and support top-K recommendations.

### What we aim to do
- **Predict ratings** \(\hat{r}(u,i)\) for unseen user–movie pairs.
- **Rank items per user** and report **Precision@K** and **Recall@K**.
- Provide a **clear, reproducible pipeline** from raw data to evaluated model, including a flow diagram of the full process.

### How we did it
1. **Data loading & basic stats**  
   Load `ratings.csv` and inspect unique users/movies to size the model.
2. **ID normalization (0-based)**  
   Use `LabelEncoder` to map `userId` and `movieId` to contiguous indices \([0..N-1]\) required by `nn.Embedding`.
3. **Train/test split**  
   80/20 with `random_state=123` for reproducibility.
4. **Dataset & DataLoaders**  
   `MovieDataset(userId, movieId, rating)` returns tensors; `DataLoader` handles batching and shuffling.
5. **Model**  
   Two embedding tables (`nn.Embedding` for users and movies). Concatenate `[user_emb; movie_emb]` and project with a linear layer to a **single rating**.
6. **Training**  
   Optimize **MSELoss** with **Adam**. Track progress with `tqdm` and plot epoch-level training loss.
7. **Evaluation**  
   Report **MSE**, **RMSE**, **MAE** on the test set; compute **Precision@K**/**Recall@K** per user (macro/micro averages).
8. **Documentation**  
   Generate a **pipeline diagram** (Matplotlib) showing each stage and how data flows (encoders → vocab sizes → model, etc.).

> Hyperparameters (e.g., `NUM_EMBEDDINGS`, `BATCH_SIZE`, `NUM_EPOCHS`) are defined at the top of the notebook and can be tuned based on GPU memory and desired accuracy.
