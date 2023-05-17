export PYTHONPATH=$PYTHONPATH:`pwd`

# d=1
python ./main_simulation_bGMM.py \
  --train_size_list 20,50,100,200,500 \
  --gamma_list 0,1,2,5,10,20,50 \
  --d_list 1 \
  --mode m

# m_S = 10
python ./main_simulation_bGMM.py \
  --train_size_list 10 \
  --gamma_list 0,1,2,5,10,20,50 \
  --d_list 2,10,20,50,100 \
  --mode d

# d = 1, m_S = 40
python ./main_simulation_bGMM.py \
  --train_size_list 40 \
  --gamma_list 0,1,2,5,10,20,50,100 \
  --d_list 1 \
  --mode gamma

# d = 50, m_S = 10
python ./main_simulation_bGMM.py \
  --train_size_list 10 \
  --gamma_list 0,1,2,5,10,20,50,100 \
  --d_list 50 \
  --mode gamma
