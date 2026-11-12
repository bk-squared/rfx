Multi-device snapshot, plane, flux, far-field and current-moment records now
join each device's owned data in bounded chunks, without first copying every
device's carry onto every device. This removes the extra full-carry copies
from the join. Each device still receives the complete returned record, so
that record must fit on one device. Signed zeros and differentiation through
the returned records are preserved.
