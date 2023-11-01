# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 12:00:11 2023

@author: Breach
"""



import matplotlib.pyplot as plt
import numpy as np

def sum_vals(episode_range,error):
    summed_vals=[]
    n0=0
    for range_set in episode_range:
        summed_vals.append(sum(error[n0:range_set]))
        n0=range_set
    return summed_vals

embeded_error=np.load("embeded_error_total.npy")
embeded_global=np.load("embeded_global_total.npy")
embeded_local=np.load("embeded_local_total.npy")

error_state=np.load("error_state.npy")
state_global=np.load("global_error_state.npy")
state_local=np.load("local_error_state.npy")

run_result=np.load("run_results.npy")
run_steps=np.load("run_steps.npy")

run_evaluation_results=np.load("run_evaluation_results.npy")
run_evaluation_steps=np.load("run_evaluation_steps.npy")

embedded_error_caution=np.load("embedded_error_caution.npy")
embedded_global_caution=np.load("embedded_global_caution.npy")
embedded_local_caution=np.load("embedded_local_caution.npy")

summed_embeded_error=sum_vals(run_steps,np.abs(embeded_error))
summed_error_stater=sum_vals(run_steps,np.abs(error_state))

summed_embeded_global=sum_vals(run_steps,np.abs(embeded_global))
summed_state_global=sum_vals(run_steps,np.abs(state_global))

summed_embeded_local=sum_vals(run_steps,np.abs(embeded_local))
summed_state_local=sum_vals(run_steps,np.abs(state_local))




def plot_arrays(x, *arrays, labels=None, title=None, xlabel=None, ylabel=None, legend=True, grid=True, figsize=(8, 6)):

    plt.figure(figsize=figsize)
    
    for i, array in enumerate(arrays):
        label = labels[i] if labels else None
        plt.plot(array, label=label)
    
    if legend:
        plt.legend()
    
    if title:
        plt.title(title)
    
    if xlabel:
        plt.xlabel(xlabel)
    
    if ylabel:
        plt.ylabel(ylabel)
    
    if grid:
        plt.grid(True)
    
    plt.show()

y=[embeded_error,error_state]
x = np.linspace(0, 1, 68425)
title="The ERROR Difference Embedded vs State"
labels=["embeded_error","state_error"]
xlabel="training steps"
ylabel="loss"

plot_arrays(x,np.abs(embeded_error),np.abs(error_state),labels=labels,title=title,xlabel=xlabel,ylabel=ylabel)


title="The Global Difference Embedded vs State"
labels=["embeded_global_error","state_global_error"]
plot_arrays(x,embeded_global,state_global,labels=labels,title=title,xlabel=xlabel,ylabel=ylabel)


title="The Local Difference Embedded vs State"
labels=["embeded_local_error","state_local_error"]
plot_arrays(x,embeded_local,state_local,labels=labels,title=title,xlabel=xlabel,ylabel=ylabel)


title="The Error Difference Embedded vs State"
labels=["embeded_error","state_error"]
plot_arrays(x,np.abs(embeded_error),np.abs(error_state),labels=labels,title=title,xlabel=xlabel,ylabel=ylabel)


title="The Reward Responce per episode at the given total timestep with random sample"
figsize=(8, 6)
plt.figure(figsize=figsize)
labels=["episodic reward"]
xlabel="steps"
ylabel="reward"
plt.plot(run_steps, run_result)
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid(True)
plt.show()

title="The Reward Responce per episode at the given total timestep"
figsize=(8, 6)
plt.figure(figsize=figsize)
labels=["episodic reward"]
xlabel="steps"
ylabel="reward"
plt.plot(run_evaluation_steps, run_evaluation_results)
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid(True)
plt.show()


title="The Reward Responce per episode with random sample with summed error"
labels=["episodic reward"]
xlabel="steps"
ylabel="reward"
figsize=(8, 6)
plt.figure(figsize=figsize)
plt.plot(run_steps, run_result,label="episodic reward")
plt.plot(run_steps, summed_embeded_error,label="summed embedded error")
plt.plot(run_steps, summed_error_stater,label="summed state error")
plt.legend(loc="lower right")
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid(True)
plt.show()



title="The Reward Responce per episode with random sample with summed global error"
xlabel="steps"
ylabel="reward"
figsize=(8, 6)
plt.figure(figsize=figsize)
plt.plot(run_steps, run_result,label="episodic reward")
plt.plot(run_steps, summed_embeded_global,label="summed embedded global")
plt.plot(run_steps, summed_state_global,label="summed state global")
plt.legend(loc="lower right")
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid(True)
plt.show()

title="The Reward Responce per episode with random sample with summed local error"
xlabel="steps"
ylabel="reward"
figsize=(8, 6)
plt.figure(figsize=figsize)
plt.plot(run_steps, run_result,label="episodic reward")
plt.plot(run_steps, summed_embeded_global,label="summed embedded local error")
plt.plot(run_steps, summed_state_global,label="summed state local error")
plt.legend(loc="lower right")
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid(True)
plt.show()


title="The Reward Responce per episode with random sample with summed local error"
xlabel="steps"
ylabel="reward"
figsize=(8, 6)
plt.figure(figsize=figsize)
plt.plot(run_steps, run_result,label="episodic reward")
plt.plot(run_steps, summed_embeded_global,label="summed embedded local error")
plt.plot(run_steps, summed_state_global,label="summed state local error")
plt.legend(loc="lower right")
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid(True)
plt.show()



title="The Global Difference Embedded vs Embeded with caution"
labels=["embeded_global_error","embeeded_global_caution"]
plot_arrays(x,embeded_global,embedded_global_caution,labels=labels,title=title,xlabel=xlabel,ylabel=ylabel)

title="The Local Difference Embedded vs Embeded with caution"
labels=["embeded_local_error","embeeded_local_caution"]
plot_arrays(x,embeded_local,embedded_local_caution,labels=labels,title=title,xlabel=xlabel,ylabel=ylabel)


title="The Error Difference Embedded vs Embeded with caution"
labels=["embeded_error","embeeded_error_caution"]
plot_arrays(x,np.abs(embeded_error),np.abs(embedded_error_caution),labels=labels,title=title,xlabel=xlabel,ylabel=ylabel)


    