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
    <img src="https://github.com/upskiller-xyz/Daylight-Factor/blob/main/docs/images/logo_upskiller.png" alt="Logo" width="80" height="80">
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
        <li><a href="#estimate-daylight-factor">Estimate Daylight Factor</a></li>
        <li><a href="#deployment">Deployment</a>
          <li><a href="#locally">Local deployment</a></li>
        </li>
    </li>
    <li><a href="#design">Design</a>
      <li><a href="#architecture">Architecture</a></li>
      <li><a href="#endpoints">Endpoints</a></li>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contribution">Contribution</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#attribution">Attribution</a></li>
    <li><a href="#trademark-notice">Trademark notice</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![image](https://github.com/upskiller-xyz/Daylight-Factor/blob/main/docs/images/heatmap_in_3d.png)

This is code for the server for Daylight factor estimation project. It can be used for local as well as for cloud deployment and usage.


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
1. [Download](https://upskiller.xyz/df_model) the model

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

See [API Documentation](./docs/api.md) for endpoint details and example requests.

### Estimate Daylight Factor

Getting results from function can be done by running [the following code](./example/demo.ipynb):

```py
import base64
import cv2
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import requests

server_url = "http://127.0.0.1:8081"
img_folder = "assets"
imgname = "image.png"

img_path = "{}/{}".format(img_folder, imgname)
im = cv2.imread(img_path)
is_success, buffer = cv2.imencode(".png", im)
img_bytes = io.BytesIO(buffer.tobytes())


files = [
("file", ("filename", img_bytes))]
resp = requests.post("{}/get_df".format(server_url), files=files, data={
        "translation":json.dumps({"x": 0, "y": 0}), 
        "rotation": json.dumps([0.0])
    })

k = resp.json()["content"]
# k = base64.b64decode(k)
img = np.load(io.BytesIO(base64.b64decode(k)))
plt.imshow(img)

```

### Deployment

* Make sure your changes are followed by an update in ```__version__.py```
* Update an ```.env``` file with the following values:
```sh
GCP_REGION=gcpregion
SERVER_NAME=servername
REPO_NAME=reponame
IMAGE_NAME=imgname
```
* Authenticate to GCP account 
```sh
gcloud auth login
```
* Deploy the new version of the server
```sh
bash build.sh
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Locally

These steps will help you set up and run the Daylight server API locally for development and testing.

1. **Clone the Repository**

  ```bash
  git clone <your-repo-url>
  cd upskiller/server
  ```
2. **Set Up a Python Virtual Environment**

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
3. **Install Dependencies**

  ```bash
  pip install -r requirements.txt
  ```
4. **Set Environment Variables (Optional)**

  You can set the server port or other environment variables as needed:

  ```bash
  export PORT=8081
  ```

5. **Run the Server**

  ```bash
  cd src
  python main.py
  ```

  The server will start on `http://localhost:8081` by default.

6. **Run Tests** - optional

  To run all tests (including those using real assets):

  ```bash
  pytest tests/
  ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DESIGN -->
## Design

### Architecture: 

![Class Diagram: Components](./docs/library.svg)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Endpoints: 

![Endpoints](./docs/api.md)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/upskiller-xyz/server/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTION -->
## Contribution

See [Contribution](./docs/CONTRIBUTING.md) for more details on contribution.

**Some guidelines:**

* Use OOP and try to follow the existing code deisgn patterns;
* We use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) - or at least try to.
* We use [semantic versioning](https://semver.org/) and tags to navigate the packages.
* Noticed something that is not working as it should? [Submit](https://github.com/upskiller-xyz/server/issues/new) an issue.



<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

See [License](./docs/LICENSE) for more details - or [read a summary](https://choosealicense.com/licenses/gpl-3.0/).

In short:

Strong copyleft. You **can** use, distribute and modify this code in both academic and commercial contexts. At the same time you **have to** keep the code open-source under the same license (`GPL-3.0`) and give the appropriate [attribution](#attribution) to the authors.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Attribution

📖 **Academic/Industry Use**: Please cite this work as described in [CITATION.cff](docs/citation/CITATION.cff), [CITE.txt](docs/citation/CITE.txt) or [ATTRIBUTION.md](docs/citation/ATTRIBUTION.md). Alternatively you can download the BibTeX file [here](docs/citation/daylight-server.bib) by adding it to `.tex` files by

```tex
\bibliography{daylight-server}
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Trademark Notice  

- **"Upskiller"** is an informal collaborative name used by contributors affiliated with BIMTech Innovations AB.  
- BIMTech Innovations AB owns all legal rights to the **Daylight Factor Estimation Server** project.  
- The GPL-3.0 license applies to code, not branding. Commercial use of the names requires permission.

Contact: [Upskiller](mailto:info@upskiller.xyz)

## Contact

Stasja Fedorova - [e-mail](mailto:stasya.fedorova@upskiller.xyz)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [README template](https://github.com/othneildrew/Best-README-Template)
* [Belysningsstiftelsen](https://belysningsstiftelsen.se/) - for financial support.
* [IAAC](https://iaac.net) - for providing the ground for preliminary study of daylight factor in architectural context.
* Hande Karataş, Dawid Drożdż, Angelos Chronis, Gabriella Rossi, Vasiliki Fragkia, Marie-Claude Dubois - for contributing with their knowledge to the preliminary study and exploration phase.

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
[license-url]: https://github.com/upskiller-xyz/server/blob/master/docs/LICENSE.txt
[product-screenshot]: https://github.com/upskiller-xyz/Daylight-Factor/blob/main/docs/images/heatmap_in_3d.png