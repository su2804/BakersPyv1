Before we start baking (analyzing baker's yeast), its important to create an environment that we can rely on to reproduce our analysis. 
BakersPy has the following main dependencies:
<ul>
<li><p>YeaZ</p></li>
<li><p>read-roi</p></li>
<li><p>napari</p></li>
<li><p>nd2reader</p></li>

</ul>
To install YeaZ, follow the instructions [here] (https://github.com/lpbsscientist/YeaZ-GUI). While installing YeaZ, the instructions will prompt you to create a
conda environment called 'YeaZ'. We will use this environment as the core of BakersPy and add/install packages based on our needs.

To clone YeaZ type the following in command line: 

<pre><code>
conda create --clone YeaZ --name YeaZ_pymitoquant
</code></pre>


Next, activate this environment using

<pre><code>
conda activate YeaZ_pymitoquant
</code></pre>

Finally, install the second dependency which is read-roi using the following command:
<pre><code>
pip install read-roi
</code></pre>

To install napari (viewing images:
<pre><code>
pip install 'napari[all]'
</code></pre>

To install nd2reader:
<pre><code>
pip install nd2reader
</code></pre>


To write scripts in this environment, just type in the name of your favorite editor (i use atom)
<pre><code>
atom
</code></pre>
