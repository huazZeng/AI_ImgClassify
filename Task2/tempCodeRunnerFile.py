def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)

        return image, label