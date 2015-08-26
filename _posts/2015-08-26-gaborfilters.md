---
layout: post
title: "Gabor Filters"
date: 2015-08-26 14:12:57
tags:
    - experiment
image: "/notebooks/images/Gabors/DTS_Photography_Movie7.jpg"
comments: true
---
 
<style>
    .center-image { display: block; margin: auto; }
</style>

# Gabor Filters

In this notebook, I will describe what Gabor Filters are and demonstrate some of their uses. 
 
# What are Gabor filters?

* Gabor filter, named after the physicist and electrical engineer Dennis Gabor, is a linear filter used for edge detection in image processing.

<img src="/notebooks/images/Gabors/Dennis_Gabor.jpg" class="image-center" style="width: 211px;"/>

(Trivia: Dennis Gabor invented holography and received Nobel Prize in 1971!)  
 
Gabor filters were originally introduced as an acoustic (1D) filter to explain audition. 

<img src="/notebooks/images/Gabors/paper.png" class="image-center" style="width: 25%;"/> 
 
A Gabor filter is a product of the *Gaussian envelope* and the (complex) *sinusoidal carrier*:

$$g(x; \sigma, f, \phi) = \underbrace{\exp\left( -\frac{x^2}{2 \sigma^2} \right)}_{\textrm{Gaussian}}
\underbrace{ \exp\left( i \left(\frac{2 \pi x}{f} + \phi \right) \right)}_{\textrm{Sinusoid}}$$

Remember, $e^{i x} = \cos x + i \sin x$; so the filter has both real and imaginary parts. 

{% highlight python %}
xs=linspace(-3, 3, 1000)
gaussian=exp(-xs**2/2)
sinusoid=exp(1j*2*pi*xs/0.5) # 1j is "i"
plt.plot(xs, gaussian, '--')
plt.plot(xs, real(sinusoid), '--')
plt.plot(xs, gaussian*real(sinusoid))
{% endhighlight %} 
![](/assets/2015-08-26-gaborfilters_files/2015-08-26-gaborfilters_5_1.png) 
 
Similarly, we can characterize 2D Gabor filters using:

 * For the Gaussian part: standard deviations $\sigma_x$ and $\sigma_y$
 * For the sinuosoid part: the frequency $f$ and phase offset $\phi$
 * Rotate the the filter with angle $\theta$ 
 
Exercise 1. Let's create some 2D Gabor filters and visualize them. 

{% highlight python %}
#from skimage.filters import gabor_kernel   
#   ^--- if you have latest version of scikit-image installed.
from filters import gabor_kernel

g=gabor_kernel(frequency=0.1, theta=pi/4, 
               sigma_x=3.0, sigma_y=5.0, offset=pi/5, n_stds=5)
plt.imshow(real(g), cmap='RdBu', interpolation='none', 
           vmin=-abs(g).max(), vmax=abs(g).max())
plt.axis('off')  
{% endhighlight %} 
![](/assets/2015-08-26-gaborfilters_files/2015-08-26-gaborfilters_8_1.png) 
 
## What do Gabor filters do?

 * Gabor filters are good **feature detectors**.
 * Each filter is sensitive to **edges** with specific **orientations** at specific **locations**.
 * However, there is a trade-off between spatial resolution and frequency resolution ("Uncertainty Principle").
   * If you have high spatial resolution (know where the edge occurs), you have less certainty in the frequency contents.
 * It turns out, mathematically, Gabor filter achieves a kind of optimal space-time trade-off.  
 
For the purposes of cognitive modeling, however, the most important is that the filter responses achieve **translation invariance**: that is, the *magnitude* of the (complex) filter response doesn't change very much when the image is shifted slightly. 
 
Exercise 2. Let's apply different Gabor filters to an image and visualize the filter responses.

(This can be done very easily using `skimage.filters.gabor_filter` function.)

Try changing the values in the sliders. Note how the magnitude of the complex response (`response = abs`) provides some *shift invariance* - that is, the response changes less even when the image is shifted slightly (`image_shift_x` and `image_shift_y`.) 


{% highlight python %}
#from skimage.filters import gabor_filter
from filters import gabor_filter
import pickle
dataset=pickle.load(open('data/cafe.pkl','r'))

interactive(plot_gabor_filtering, 
            dataset=fixed(dataset),
            image_id=(0,dataset.images.shape[0]-1),
            frequency=FloatSlider(min=0.001, max=0.5, value=0.1),
            theta=FloatSlider(min=0, max=pi, value=pi / 2.0),
            response_fn={'Real': real, 'Imag': imag, 'Abs': abs},
            image_shift_x=IntSlider(min=0, max=10, value=0),
            image_shift_y=IntSlider(min=0, max=10, value=0))
{% endhighlight %} 
![](/assets/2015-08-26-gaborfilters_files/2015-08-26-gaborfilters_13_0.png) 
 
## But why do we use Gabor filters for cognitive modeling?

* It turns out Gabor filters are good model for capturing the *statistics* of natural images.
* To the extent that our sensory apparatuses have evolved to efficiently encode the statistics of the world we live in (Barlow’s "efficient coding hypothesis"), [the statistics of natural images reveal the workings of our perceptual system](https://courses.cs.washington.edu/courses/cse528/11sp/Olshausen-nature-paper.pdf). 

[Download this IPython notebook]()
