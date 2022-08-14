# TSFool: Crafting High-quality Adversarial Time Series to Fool Recurrent Neural Network Classifiers

## Update

#### August 2022
- The raw experiment records including **1) the datasets selected from the UCR archive, 2) the basic parameters of used LSTM classifiers, 3) intervalized weighted finite automatons established in the process, and 4) the results of the final adversarial attacks** have been opened now.
- AA

#### July 2022
- The sample program using TSFool to craft adversarial time series for an LSTM classifier in PowerCons Dataset from UCR Archive is added for reference in advance.
- The work is in progress at present and the detailed description (as well as a possible technology paper) will be opened to the public soon.


## Core Idea

One of the possible explanations for the existence of the adversarial sample is that, the features of the input data cannot always fully and visually reflect the latent manifold, which makes it possible for samples that are considered to be similar in the external features to have radically different latent manifolds, and as a result, to be understood and processed in a different way by the NN. Therefore, even a small perturbation in human cognition imposed on the correct sample may completely overturn the NN's view of its latent manifold, so as to result in a completely different result. 

So if there is a kind of model that can simulate the way an NN understands and processes input data, but distinguish different inputs by their original features in the high-dimensional space just like a human, then it can be used to capture the otherness between the latent manifold and external features of any input sample. And such otherness can serve as guidance to find the potential vulnerable samples for the adversarial attack to improve its success rate, efficiency and quality.

In this project, Interval Weighted Finite Automaton and Recurrent Neural Network (actually LSTM) are respectively the model and the NN mentioned above.


## The Experimental Datasets Selected from UCR Archive

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


## The Experimental LSTM Classifiers

<table class="MsoNormalTable" border="0" cellspacing="0" cellpadding="0" width="529" style="width:396.65pt;border-collapse:collapse;mso-yfti-tbllook:1184;
 mso-padding-alt:0cm 5.4pt 0cm 5.4pt">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Abbrev ID</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="94" style="width:70.85pt;border:solid windowtext 1.0pt;border-left:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Model ID</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Loss</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border:solid windowtext 1.0pt;
  border-left:none;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;background:
  #D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Train Accuracy</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border:solid windowtext 1.0pt;
  border-left:none;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;background:
  #D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Test Accuracy</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">CBF<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E200B0<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.504382<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7667<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7511<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">DPOAG<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E199B0<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.315647<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8525<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7842<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">DPOC<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E117B2<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.574497<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7767<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7319<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">DPTW<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E112B1<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.440651<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8150<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7122<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">ECG2<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E179B0<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.242467<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8800<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7900<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:6;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">ECG5<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E69B2<sup>*</sup><o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.179686<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9480<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9267<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:7;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">GP<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E64B0<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.226873<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt">0.9400<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt">0.9333<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:8;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">IPD<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E88B0<sup>*<o:p></o:p></sup></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.043879<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9701<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9650<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:9;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">MPOAG<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E163B1<sup>*</sup><o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.442033<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7875<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6429<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:10;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">MPOC<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E200B2<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.581016<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7017<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7457<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:11;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">MPTW<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E97B0<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.836946<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6767<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6169<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:12;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">PPOAG<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E173B0<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.404579<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8200<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8976<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:13;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">PPOC<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E195B0<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.447616<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7800<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7869<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:14;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">PPTW<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E68B1<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.551807<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7775<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8000<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:15;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">SC<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E43B1<sup>*</sup><o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.245713<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9400<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9400<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:16;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">TP<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E78B4<sup>*</sup><o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.008251<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9990<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9993<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:17;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">GPAS<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E121B0<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.588720<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8000<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8418<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:18;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">GPMVF<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E200B0<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.184646<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9630<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9652<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:19;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">GPOVY<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E13B0<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.102776<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9926<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9778<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:20;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">PC<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E39B0<sup>*</sup><o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.141841<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9500<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9444<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:21;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">SS<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E126B0<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.190155<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt">0.9267</span><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt"><pre style="text-align:center;
  line-height:12.0pt;tab-stops:21.0pt"><span lang="EN-US" style="font-size:10.5pt;
  font-family:&quot;Times New Roman&quot;,serif">0.9133<o:p></o:p></span></pre></td>
 </tr>
 <tr style="mso-yfti-irow:22;mso-yfti-lastrow:yes;height:21.25pt">
  <td width="94" style="width:70.65pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">UMD<o:p></o:p></span></p>
  </td>
  <td width="94" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">E187B0<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.347076<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8056<o:p></o:p></span></p>
  </td>
  <td width="113" nowrap="" style="width:3.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7847<o:p></o:p></span></p>
  </td>
 </tr>
</tbody></table>


## The Intervalized Weighted Finite Automatons Established

<table class="MsoNormalTable" border="0" cellspacing="0" cellpadding="0" width="576" style="width:432.1pt;border-collapse:collapse;mso-yfti-tbllook:1184;
 mso-padding-alt:0cm 5.4pt 0cm 5.4pt">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Abbrev ID</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="57" style="width:42.55pt;border:solid windowtext 1.0pt;border-left:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">k</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="66" style="width:49.6pt;border:solid windowtext 1.0pt;border-left:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">t</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="66" style="width:49.6pt;border:solid windowtext 1.0pt;border-left:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span class="SpellE"><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;color:black;mso-color-alt:windowtext;
  mso-font-kerning:0pt">r<sub>input</sub></span></b></span><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">|Z|</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;background:
  #D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">|S|</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;background:
  #D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Accuracy</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="85" style="width:63.8pt;border:solid windowtext 1.0pt;border-left:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Similarity</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">CBF<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">3<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">15<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.15<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">130<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">184<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7667<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">93.33%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">DPOAG<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">3<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">30<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.07<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">858<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">685<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7200<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">76.00%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">DPOC<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">700<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.05<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1123<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">518<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5483<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">70.50%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">DPTW<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">3<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">15<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.2<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">311<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">747<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6800<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">76.00%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">ECG2<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">300<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.1<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">413<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">300<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7000<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">74.00%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:6;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">ECG5<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">3<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">20<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">120<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">989<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8820<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">92.60%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:7;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">GP<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">200<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.3<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">761<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt">185<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8600<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">88.00%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:8;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">IPD<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">100<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.15<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">175<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">97<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7761<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">77.61%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:9;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">MPOAG<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">3<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">35<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.1<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">699<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">718<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7475<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">88.50%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:10;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">MPOC<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">300<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.2<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">351<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">263<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6683<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">95.33%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:11;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">MPTW<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">3<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">22<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.375<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">186<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1215<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.4812<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">56.89%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:12;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">PPOAG<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">3<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">25<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.03<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2156<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">537<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7100<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">79.25%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:13;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">PPOC<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">400<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.1<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">692<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">389<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7300<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">76.00%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:14;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">PPTW<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">3<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">20<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">135<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">713<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7725<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">94.00%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:15;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">SC<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">3<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">10<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.04<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">352<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">368<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7533<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">78.67%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:16;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">TP<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">3<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">13<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.2<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">151<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">627<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6950<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">69.40%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:17;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">GPAS<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">700<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.01<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">171<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">451<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7185<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">88.89%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:18;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">GPMVF<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">500<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.005<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">344<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">499<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8000<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">79.26%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:19;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">GPOVY<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">300<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.02<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">88<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">265<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9632<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">95.59%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:20;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">PC<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">150<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">104<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">149<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8000<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">82.78%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:21;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">SS<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">3<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">15<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.4<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">94<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt"><pre style="text-align:center;
  line-height:12.0pt;tab-stops:21.0pt"><span lang="EN-US" style="font-size:10.5pt;
  font-family:&quot;Times New Roman&quot;,serif">188<o:p></o:p></span></pre></td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8000<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">84.00%<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:22;mso-yfti-lastrow:yes;height:21.25pt">
  <td width="85" style="width:63.55pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">UMD<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">3<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">60<o:p></o:p></span></p>
  </td>
  <td width="66" style="width:49.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">347<o:p></o:p></span></p>
  </td>
  <td width="66" nowrap="" style="width:49.65pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">370<o:p></o:p></span></p>
  </td>
  <td width="85" nowrap="" style="width:63.75pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6389<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">77.78%<o:p></o:p></span></p>
  </td>
 </tr>
</tbody></table>


## The Results of Adversarial Time Series Crafting

<table class="MsoNormalTable" border="0" cellspacing="0" cellpadding="0" width="548" style="border-collapse:collapse;mso-table-layout-alt:fixed;mso-yfti-tbllook:
 1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes;height:21.25pt;mso-height-rule:
  exactly">
  <td width="87" rowspan="2" style="width:65.4pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Abbrev ID</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="56" rowspan="2" style="width:41.9pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt"><span class="SpellE"><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;color:black;
  mso-color-alt:windowtext;mso-font-kerning:0pt">f<sub>ampt</sub></span></b></span><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="57" rowspan="2" style="width:42.5pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt"><span class="SpellE"><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;color:black;
  mso-color-alt:windowtext;mso-font-kerning:0pt">f<sub>win</sub></span></b></span><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="83" rowspan="2" style="width:62.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Samples </span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Generated</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="94" nowrap="" rowspan="2" style="width:70.85pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Adverse Rate</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="170" colspan="2" style="width:127.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;background:
  #D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">RNN Accuracy</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1;height:21.25pt;mso-height-rule:exactly">
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;background:
  #D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Original</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
  <td width="85" style="width:63.8pt;border:solid windowtext 1.0pt;border-left:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  background:#D6DCE4;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><b><span lang="EN-US" style="mso-bidi-font-size:
  10.5pt;color:black;mso-color-alt:windowtext;mso-font-kerning:0pt">Attacked</span></b><b><span lang="EN-US" style="mso-bidi-font-size:10.5pt;mso-font-kerning:0pt"><o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">CBF<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">234<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">37.61%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7511<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7249<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">DPOAG<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2926<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">32.02%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7842<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6845<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">DPOC<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2920<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">41.92%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7319<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5904<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">DPTW<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">146<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">28.08%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7122<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6842<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:6;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">ECG2<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">704<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">46.59%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7900<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5659<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:7;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">ECG5<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">254<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">46.85%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9267<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9056<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:8;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">GP<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">272<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">25.37%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9333<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8033<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:9;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">IPD<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">48<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">47.92%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9650<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9424<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:10;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">MPOAG<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1314<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">47.34%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6429<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5272<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:11;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">MPOC<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">584<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">49.32%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7457<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5863<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:12;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">MPTW<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">146<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">35.62%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6169<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6033<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:13;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">PPOAG<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">2336<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;
  mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">29.97%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8976<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7131<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:14;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">PPOC<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">8322<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;
  mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">33.38%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7869<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6677<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:15;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">PPTW<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1314<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">35.39%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8000<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6570<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:16;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">SC<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">330<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">49.70%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9400<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7111<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:17;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">TP<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">234<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;
  mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">41.03%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9993<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9766<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:18;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">GPAS<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">544<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;
  mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">40.62%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8418<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6349<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:19;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">GPMVF<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">816<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">49.88%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9652<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6307<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:20;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">GPOVY<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">272<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">38.97%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9778<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7445<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:21;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">PC<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">524<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;
  mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">20.42%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9444<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.8281<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:22;height:21.25pt;mso-height-rule:exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">SS<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.5<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">60<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;
  mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">43.33%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.9133<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7905<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:23;mso-yfti-lastrow:yes;height:21.25pt;mso-height-rule:
  exactly">
  <td width="87" style="width:65.4pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">UMD<o:p></o:p></span></p>
  </td>
  <td width="56" style="width:41.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">1<o:p></o:p></span></p>
  </td>
  <td width="83" style="width:62.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">272<o:p></o:p></span></p>
  </td>
  <td width="94" nowrap="" style="width:70.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">37.87%<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.7847<o:p></o:p></span></p>
  </td>
  <td width="85" style="width:63.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:21.25pt;mso-height-rule:
  exactly">
  <p class="MsoNormal" align="center" style="text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan"><span lang="EN-US" style="mso-bidi-font-size:10.5pt;
  mso-font-kerning:0pt">0.6731<o:p></o:p></span></p>
  </td>
 </tr>
</tbody></table>

