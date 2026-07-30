function xRef = commonMedianRef(xRaw)
%COMMONMEDIANREF Subtract the per-sample median across channels.
%
%   xRaw: samples x channels. This step was already causal in the offline
%   pipeline (it only ever uses the current sample's 64 channel values,
%   never neighboring time points), so it carries over unchanged to the
%   real-time version -- no redesign needed here.

    xRef = xRaw - median(xRaw, 2);
end
