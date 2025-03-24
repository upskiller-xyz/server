<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/upskiller-xyz/server">
    <img src="../assets/.docs/logo.svg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center" Store Visibility </h3>

  <p align="center">
    Daylight Server
    <br />
    <a href="https://github.com/upskiller-xyz/server">View Demo</a>
    ·
    <a href="https://github.com/upskiller-xyz/server/issues">Report Bug</a>
    ·
    <a href="https://github.com/upskiller-xyz/server/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
        <li><a href="#deployment">Usage</a></li>
    </li>
    <li><a href="#design">Design</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This is the source code for Upskiller's server that makes daylight factor predictions available to the wider public.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [python](https://www.python.org/)
* [flask](https://flask.palletsprojects.com/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites


* [python3](https://www.python.org/downloads/)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/upskiller-xyz/server.git
   ```
1. Create a virtual environment and install dependencies:
```sh
  python -m venv server
  server/Scripts.activate
  pip install -r requirements.txt
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Predict

Getting results from function can be done by running [the following code](./demo.ipynb):

```py
import requests
import numpy as np

url = "https://daylight-server-182483330095.europe-north2.run.app/daylight_factor"
k = requests.post(url, json={"content": {"image": inp.tolist()}})
r = k.json()  # r.status_code==200
image = np.array(r["content"][0])
n_image = (image + 1) * 127.5

```

### Deployment

* Make sure your changes are followed by an update in ```__version__.py```
* Authenticate to GCP account (```daylight-factor``` project)
```sh
gcloud auth login
```
* Deploy the new version of the server
```sh
bash build.sh
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DESIGN -->
## Design

### Library: 

![Class Diagram: Components](../assets/.docs/library.svg)

### Endpoints: 

![Endpoints](../assets/.docs/endpoints.svg)

<!-- ROADMAP -->
## Roadmap

- [ ] Add CI/CD
- [ ] Add different model versions

See the [open issues](https://github.com/upskiller-xyz/server/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

H&M internal project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contact

Stasja Fedorova - [e-mail](mailto:stasya.fedorova@upskiller.xyz)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [README template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/upskiller-xyz/server.svg?style=for-the-badge
[contributors-url]: https://github.com/upskiller-xyz/server/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/upskiller-xyz/server.svg?style=for-the-badge
[forks-url]: https://github.com/upskiller-xyz/server/network/members
[stars-shield]: https://img.shields.io/github/stars/upskiller-xyz/server.svg?style=for-the-badge
[stars-url]: https://github.com/upskiller-xyz/server/stargazers
[issues-shield]: https://img.shields.io/github/issues/upskiller-xyz/server.svg?style=for-the-badge
[issues-url]: https://github.com/upskiller-xyz/server/issues
[license-shield]: https://img.shields.io/github/license/upskiller-xyz/server.svg?style=for-the-badge
[license-url]: https://github.com/upskiller-xyz/server/blob/master/LICENSE.txt
[product-screenshot]: assets/screenshot.png