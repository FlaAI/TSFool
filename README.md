# TSFool: Crafting High-quality Adversarial Time Series to Fool Recurrent Neural Network Classifiers

July 2022 update: 

- The sample program using TSFool to craft adversarial time series for an LSTM classifier in PowerCons Dataset from UCR Archive is added for reference in advance.

- The work is in progress at present and the detailed description (as well as a possible technology paper) will be opened to the public soon.


## Core Idea

One of the possible explanations for the existence of the adversarial sample is that, the features of the input data cannot always fully and visually reflect the latent manifold, which makes it possible for samples that are considered to be similar in the external features to have radically different latent manifolds, and as a result, to be understood and processed in a different way by the NN. Therefore, even a small perturbation in human cognition imposed on the correct sample may completely overturn the NN's view of its latent manifold, so as to result in a completely different result. 

So if there is a kind of model that can simulate the way an NN understands and processes input data, but distinguish different inputs by their original features in the high-dimensional space just like a human, then it can be used to capture the otherness between the latent manifold and external features of any input sample. And such otherness can serve as guidance to find the potential vulnerable samples for the adversarial attack to improve its success rate, efficiency and quality.

In this project, Interval Weighted Finite Automaton and Recurrent Neural Network (actually LSTM) are respectively the model and the NN mentioned above.


## A

<table class="MsoNormalTable" border="1" cellspacing="0" cellpadding="0" width="100%" style="width:100.0%;border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt;mso-border-insideh:
 .5pt solid windowtext;mso-border-insidev:.5pt solid windowtext">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Abbrev ID</span></b></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Name</span></b></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border:solid windowtext 1.0pt;border-left:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Type</span></b></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Train </span></b></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Test </span></b></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Class</span></b></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Length</span></b></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:1;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">CBF<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">CBF<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Simulated<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">30<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">900<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">3<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">128<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:2;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">DPOAG<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">DistalPhalanxOutlineAgeGroup</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Image<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">400<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">139<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">3<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">80<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:3;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">DPOC<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">DistalPhalanxOutlineCorrect</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Image<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">600<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">276<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">2<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">80<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:4;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">DPTW<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">DistalPhalanxTW</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Image<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">400<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">139<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">6<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">80<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:5;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">ECG2<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">ECG200<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">ECG<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">100<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">100<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">2<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">96<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:6;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">ECG5<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">ECG5000<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">ECG<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">500<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">4500<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">5<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">140<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:7;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">GP<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">GunPoint</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Motion<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">50<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">150<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">2<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">150<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:8;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">IPD<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">ItalyPowerDemand</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Sensor<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">67<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">1029<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">2<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">24<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:9;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">MPOAG<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">MiddlePhalanxOutlineAgeGroup</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Image<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">400<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">154<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">3<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">80<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:10;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">MPOC<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">MiddlePhalanxOutlineCorrect</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Image<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">600<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">291<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">2<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">80<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:11;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">MPTW<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">MiddlePhalanxTW</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Image<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">399<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">154<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">6<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">80<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:12;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">PPOAG<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">ProximalPhalanxOutlineAgeGroup</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Image<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">400<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">205<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">3<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">80<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:13;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">PPOC<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">ProximalPhalanxOutlineCorrect</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Image<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">600<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">291<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">2<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">80<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:14;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">PPTW<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">ProximalPhalanxTW</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Image<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">400<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">205<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">6<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">80<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:15;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">SC<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">SyntheticControl</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Simulated<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">300<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">300<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">6<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">60<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:16;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">TP<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">TwoPatterns</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Simulated<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">1000<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">4000<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">4<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">128<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:17;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">GPAS<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">GunPointAgeSpan</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Motion<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">135<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">316<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">2<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">150<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:18;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">GPMVF<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">GunPointMaleVersusFemale</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Motion<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">135<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">316<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">2<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">150<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:19;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">GPOVY<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">GunPointOldVersusYoung</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Motion<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">136<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">315<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">2<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">150<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:20;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">PC<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">PowerCons</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Power<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">180<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">180<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">2<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">144<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:21;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">SS<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span class="SpellE"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt">SmoothSubspace</span></span></span></span></span></span><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Simulated<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">150<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">150<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">3<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">15<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
 <tr style="mso-yfti-irow:22;mso-yfti-lastrow:yes;height:21.25pt">
  <td width="13%" style="width:13.94%;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">UMD<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="36%" nowrap="" style="width:36.22%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="left" style="text-align:left;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">UMD<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="12%" style="width:12.44%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">Simulated<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">36<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="9%" nowrap="" style="width:9.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">144<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="7%" nowrap="" style="width:7.92%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">3<o:p></o:p></span></span></span></span></span></p>
  </td>
  
  <td width="10%" nowrap="" style="width:10.86%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span style="mso-bookmark:_Toc136438765"><span style="mso-bookmark:_Toc136438662"><span style="mso-bookmark:_Toc135817016"><span style="mso-bookmark:_Toc135816543"><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;mso-font-kerning:0pt">150<o:p></o:p></span></span></span></span></span></p>
  </td>
  
 </tr>
</tbody></table>

