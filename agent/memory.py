from torchrl.data.replay_buffers import ListStorage, ReplayBuffer
from collections import deque
from pathlib import Path
import numpy as np 
import random
import h5py

import settings

# class ReplayMemory:
#     def __init__(self, max_size=settings.MAX_MEMORY, save_path='/media/sda2/replay_memory.h5'):
#         print(f"ReplayMemory.__init__ 시작: max_size={max_size}, save_path={save_path}")
        
#         # 디렉토리 확인
#         import os
#         print(f"/media/sda2 존재? {os.path.exists('/media/sda2')}")
#         print(f"/media/sda2 쓰기 가능? {os.access('/media/sda2', os.W_OK)}")

#         self.max_size = max_size
#         self.position = 0  # 현재 저장 위치
#         self.size = 0      # 현재 저장된 데이터 수
#         self.path = Path(save_path)

#         print(f"디렉토리 생성 시도: {self.path.parent}")
#         self.path.parent.mkdir(parents=True, exist_ok=True)
#         print(f"디렉토리 생성 완료")

#         state_shape = (4, 84, 84)

#         print(f"HDF5 파일 생성 시도: {self.path}")
#         print(f"파일 존재 여부: {self.path.exists()}")

#         # HDF5 파일 생성 (데이터셋 미리 할당)
#         with h5py.File(self.path, 'w') as f:

#             print("HDF5 파일 열림, 데이터셋 생성 중...")
#             print("states 데이터셋 생성 중...")

#             # 상태(state) 저장 공간 - 상태 크기에 맞게 shape 조정 필요
#             f.create_dataset('states', 
#                            shape=(max_size, *state_shape),  # state_shape은 실제 환경에 맞게 설정
#                            dtype='float32',
#                            chunks=True,
#                            maxshape=(max_size, *state_shape))
            
#             # 행동(action) 저장 공간
#             f.create_dataset('actions',
#                            shape=(max_size,),
#                            dtype='int32',
#                            chunks=True,
#                            maxshape=(max_size,))
            
#             # 보상(reward) 저장 공간
#             f.create_dataset('rewards',
#                            shape=(max_size,),
#                            dtype='float32',
#                            chunks=True,
#                            maxshape=(max_size,))
            
#             # 다음 상태(next_state) 저장 공간
#             f.create_dataset('next_states',
#                            shape=(max_size, *state_shape),
#                            dtype='float32',
#                            chunks=True,
#                            maxshape=(max_size, *state_shape))
            
#             # 종료 여부(done) 저장 공간
#             f.create_dataset('dones',
#                            shape=(max_size,),
#                            dtype='bool',
#                            chunks=True,
#                            maxshape=(max_size,))
            
#             print("ReplayMemory 초기화 완료!")
    
#     def store(self, experience):
#         """경험 저장 (FIFO 방식)"""
#         state, action, reward, next_state, done = experience
        
#         with h5py.File(self.path, 'a') as f:
#             # 현재 위치에 데이터 저장
#             f['states'][self.position] = state
#             f['actions'][self.position] = action
#             f['rewards'][self.position] = reward
#             f['next_states'][self.position] = next_state
#             f['dones'][self.position] = done
        
#         # 다음 위치로 이동 (원형 버퍼)
#         self.position = (self.position + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
    
#     def sample(self, batch_size):
#         """무작위 샘플링"""
#         indices = np.random.choice(self.size, batch_size, replace=False)
        
#         with h5py.File(self.path, 'r') as f:
#             states = f['states'][indices]
#             actions = f['actions'][indices]
#             rewards = f['rewards'][indices]
#             next_states = f['next_states'][indices]
#             dones = f['dones'][indices]
        
#         return (states, actions, rewards, next_states, dones)
    
#     def __len__(self):
#         return self.size


class ReplayMemory:
    def __init__(self):
        self.memory = deque(maxlen=settings.MAX_MEMORY)
    
    def store(self, experience):
        self.memory.append(experience)
    
    def get_samples(self):
        return random.sample(self.memory, settings.BATCH_SIZE)
    
    def __len__(self):
        return len(self.memory)