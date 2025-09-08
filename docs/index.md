# Introduction

This repository provides a library for defining differentially private speaker anonymization systems using existing voice control models. The approach works for any voice control system that separates utterance information into constant-length speaker information (e.g. a speaker embedding) and time-varying content information (e.g. semantic features).

## Overview

The DPVC library provides a framework for performing differentially private speaker anonymization. The library's architecture is designed around a standardized "wrapper" for a voice control system that exposes methods for extracting speaker embeddings and performing inference. The library provides methods for performing the anonymization using a wrapped system, and also utilities for training the autoencoder used during anonymization.

## Installation

Install the library by cloning the repository and then running:

```
pip install .
```
