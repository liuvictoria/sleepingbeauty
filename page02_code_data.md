---
layout: page
title: Code and Data
img: sleepingbeautycastle.png # Add image post (optional)
caption: "Frank Armitage, Disney castle"
permalink: code
sidebar: true
---

---

{% if site.data.code %}
## Code
{% for script in site.data.code %}
* [**{{script.name}}**]({{site.url}}/{{site.baseurl}}/software/{{script.name}})
  \| {{script.desc}}
{% endfor %}
{% endif %}

{% if site.data.datasets %}
## Data Sets
{% for ds in site.data.datasets %}
* [{{ds.name}}]({%if ds.storage !=
  'remote'%}{{site.url}}/{{site.baseurl}}/datasets/{{ds.link}}{%
  else%}{{site.link}}{% endif %}) \| {% if ds.filetype %}(filetype:
  {{ds.filetype}}){%endif%}{% if ds.filesize %}({{ds.filesize}}){%endif%}{%
  if ds.storage == remote %} DOI: {{ds.DOI}}{%endif%}
{% endfor %}
{% endif %}

