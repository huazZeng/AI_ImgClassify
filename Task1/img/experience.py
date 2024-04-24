result_layersize = '_'.join(str(size) for size in self.layer_size)
result_fuc='_'.join(self.func)
result_lrup='_'+str(self.lr_update)
plt.savefig('Task1\img\experiencedata\\'+result_fuc+result_layersize+result_fuc+'_'+str(count)+'.png')