import pywinusb.hid as hid

all_hids = hid.find_all_hid_devices()
for idx, device in enumerate(all_hids):
    print(f"[{idx}] Vendor: {device.vendor_name}, Product: {device.product_name}, "
          f"VID: {hex(device.vendor_id)}, PID: {hex(device.product_id)}, "
          f"Usage Page: {hex(device.hid_caps.usage_page)}, Usage: {hex(device.hid_caps.usage)}")