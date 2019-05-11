# imagesearch

```python
>>> from image_search import ImageSearch
>>> search = ImageSearch()
>>> search.add_directories('../example_images/', recursive=True)
>>> print('\n'.join(search.images.keys()))
../example_images/cats/lily2.jpg
../example_images/cats/meoow.jpg
../example_images/cats/lily.jpg
../example_images/cats/meow.jpg
../example_images/definitely_not_cats/danny.jpeg
../example_images/definitely_not_cats/jon-snow.jpeg
../example_images/cats/mow.png
```

## Cats are similar to cats
```python

>>> for similarity, path in search.find_similar('../example_images/cats/lily.jpg'):
...     print(f'Similarity: {similarity},\tPath: {path}')
... 
Similarity: 19.969846725463867,    Path: ../example_images/cats/lily2.jpg
Similarity: 25.728940963745117,    Path: ../example_images/cats/meoow.jpg
Similarity: 25.87383270263672,     Path: ../example_images/cats/meow.jpg
Similarity: 26.700647354125977,    Path: ../example_images/cats/mow.png
Similarity: 31.068445205688477,    Path: ../example_images/definitely_not_cats/danny.jpeg
Similarity: 31.083133697509766,    Path: ../example_images/definitely_not_cats/jon-snow.jpeg
```

## People are similar to people

```python
>>> for similarity, path in search.find_similar('../example_images/definitely_not_cats/jon-snow.jpeg'):
...     print(f'Similarity: {similarity},\tPath: {path}')
... 
Similarity: 24.367565155029297,    Path: ../example_images/definitely_not_cats/danny.jpeg
Similarity: 30.542598724365234,    Path: ../example_images/cats/lily2.jpg
Similarity: 30.802589416503906,    Path: ../example_images/cats/mow.png
Similarity: 31.083133697509766,    Path: ../example_images/cats/lily.jpg
Similarity: 32.15031814575195,     Path: ../example_images/cats/meow.jpg
Similarity: 33.20389938354492,     Path: ../example_images/cats/meoow.jpg
```