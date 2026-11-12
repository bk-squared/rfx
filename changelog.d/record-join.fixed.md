Multi-device snapshot, plane, flux, far-field and current-moment records now
join each device's owned data in bounded chunks, without first copying every
device's carry onto every device. On two devices, this removes approximately
1.25–3 returned-record sizes of join buffer memory per device.
Each device still receives the complete returned record, so
that record must fit on one device. Signed zeros and differentiation through
the returned records are preserved.
